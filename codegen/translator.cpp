#include <cassert>
#include <fstream>

#include "translator.hpp"
#include "util.hpp"
#include "parser.hpp"

// Tab character used in translation (4 spaces)
//
#define TAB "    "

#define AUTOGENERATED_FILE_MESSAGE "// This is an automatically generated file.\n// Do not edit directly.\n//\n"
#define LINE_SEPARATOR_COMMENT "//------------------------------------------------------------------------\n"

// Flag that is set when we encounter .db $2c (BIT instruction)
// This sabotages the next instruction, so we need to translate these into goto's that skip the next instruction
//
static bool skipNextInstruction = false;
static int skipNextInstructionIndex = 0;

Translator::Translator(const std::string& inputFilename, RootNode* astRootNode) :
    inputFilename(inputFilename),
    root(astRootNode)
{
    returnLabelIndex = 0;

    sourceOutput << AUTOGENERATED_FILE_MESSAGE;
    constantHeaderOutput << AUTOGENERATED_FILE_MESSAGE;
    dataHeaderOutput << AUTOGENERATED_FILE_MESSAGE;

    translate();
}

std::string Translator::getConstantHeaderOutput() const
{
    return constantHeaderOutput.str();
}

std::string Translator::getDataHeaderOutput() const
{
    return dataHeaderOutput.str();
}

std::string Translator::getSourceOutput() const
{
    return sourceOutput.str();
}

void Translator::classifyLabels()
{
    for (std::list<AstNode*>::iterator it = root->children.begin();
         it != root->children.end(); ++it)
    {
        AstNode* node = (*it);

        // Skip non-labels
        //
        if (node->type != AST_LABEL)
        {
            continue;
        }

        LabelNode* label = static_cast<LabelNode*>(node);

        while (true)
        {
            // Check the first element of the label
            //
            AstNode* child = label->child;
            
            if (child->type == AST_LABEL)
            {
                // Nested... classify the label as an alias, and continue trying to classify the child
                //
                label->labelType = LABEL_ALIAS;
                label = static_cast<LabelNode*>(label->child);

                // Modify the AST
                // Pull the nested label out and promote to a child of the root node
                // (this makes processing simpler later)
                //
                // Instead of this:
                // (root)
                // |
                // --...
                // --(label)
                // | |
                // | --(label2)
                // --...
                //
                // We want this:
                // (root)
                // |
                // --...
                // --(label)
                // --(label2)
                // --...
                //
                ++it;
                root->children.insert(it, child);

                // We have to go back one so that the iterator points to the child
                //
                --it;
            }
            else
            {
                break;
            }
        }

        // Check the first child node
        //
        AstNode* child = label->child;
        
        // This should be a list
        //
        assert(child->type == AST_LIST);

        // Check the type of the first contained list item
        //
        ListNode* list = static_cast<ListNode*>(child);
        AstNode* listElement = list->value.node;
        assert(listElement != NULL);
        if (listElement->type == AST_INSTRUCTION)
        {
            // Code
            //
            label->labelType = LABEL_CODE;
        }
        else if (listElement->type == AST_DATA8)
        {
            label->labelType = LABEL_DATA;
        }
        else
        {
            // Should be impossible...
            //
            assert(false);
        }
    }
}

void Translator::generateCode()
{
    sourceOutput <<
        "void SMBEngine::code(int mode)\n" <<
        "{\n" <<
        TAB << "switch (mode)\n" <<
        TAB << "{\n" <<
        TAB << "case 0:\n" <<
        TAB << TAB << "loadConstantData();\n" <<
        TAB << TAB << "goto Start;\n" <<
        TAB << "case 1:\n" <<
        TAB << TAB << "goto NonMaskableInterrupt;\n" <<
        TAB << "}\n\n";
    
    // Search through the root node, and grab all code label nodes
    //
    for (std::list<AstNode*>::iterator it = root->children.begin();
         it != root->children.end(); ++it)
    {
        AstNode* node = (*it);

        if (node->type != AST_LABEL)
        {
            continue;
        }

        LabelNode* label = static_cast<LabelNode*>(node);
        if (label->labelType != LABEL_CODE)
        {
            continue;
        }

        // Output a C++ label for the label
        //
        sourceOutput << "\n";
        sourceOutput << label->value.s;

        // Output a comment, if the label has one
        //
        if (label->lineNumber != 0)
        {
            const char* comment = lookupComment(label->lineNumber);
            if (comment)
            {
                // Skip the first character of the ASM comment (;)
                //
                sourceOutput << " // " << (comment + 1);
            }
        }

        sourceOutput << "\n";

        // Translate each piece of code under the label...
        //
        ListNode* listElement = static_cast<ListNode*>(label->child);
        while (listElement != NULL)
        {
            AstNode* instruction = listElement->value.node;
            if (instruction->type == AST_INSTRUCTION)
            {
                sourceOutput << TAB << translateInstruction(static_cast<InstructionNode*>(instruction));

                if (instruction->lineNumber != 0)
                {
                    const char* comment = lookupComment(instruction->lineNumber);
                    if (comment)
                    {
                        // Skip the first character of the ASM comment (;)
                        //
                        sourceOutput << " // " << (comment + 1);
                    }
                }

                // Add a nice line separator after return statements
                //
                if (static_cast<InstructionNode*>(instruction)->code == RTS)
                {
                    sourceOutput << "\n\n";
                    sourceOutput << LINE_SEPARATOR_COMMENT;
                }
                else
                {
                    // Or just a newline for all other instructions
                    //
                    sourceOutput << "\n";
                }
                
                if (skipNextInstruction)
                {
                    // If we had a .db $2c instruction immediately before this one,
                    // we need to add a label to be able to skip this instruction
                    //
                    char indexStr[8];
                    sprintf(indexStr, "%d", skipNextInstructionIndex++);
                    sourceOutput << "Skip_" << indexStr << ":\n";

                    skipNextInstruction = false;
                }
            }
            else if (instruction->type == AST_DATA8 &&
                    strcmp(instruction->value.node->value.node->value.s, "$2c") == 0)
            {
                // Special case: .db $2c
                // We need to goto the next instruction
                //
                skipNextInstruction = true;
                char indexStr[8];
                sprintf(indexStr, "%d", skipNextInstructionIndex);
                sourceOutput << TAB << "goto Skip_" << indexStr << ";\n";
            }

            listElement = static_cast<ListNode*>(listElement->next);
        }
    }

    // Generate a return jump table at the end of the code
    //
    sourceOutput << 
        "// Return handler\n" <<
        "// This emulates the RTS instruction using a generated jump table\n" <<
        "//\n" <<
        "Return:\n" <<
        TAB << "switch (popReturnIndex())\n" <<
        TAB << "{\n";
    
    for (int i = 0; i < returnLabelIndex; i++)
    {
        char indexStr[8];
        sprintf(indexStr, "%d", i);
        sourceOutput <<
            TAB << "case " << indexStr << ":\n" <<
            TAB << TAB << "goto Return_" << indexStr << ";\n";
    }
    
    sourceOutput <<
        TAB << "}\n";
    
    // Final closing block for the code() function
    //
    sourceOutput << "}\n";
}

void Translator::generateConstantDeclarations()
{
    constantHeaderOutput <<
        "#ifndef SMBCONSTANTS_HPP\n" <<
        "#define SMBCONSTANTS_HPP\n\n";

    // Search through the root node, and grab all Decl nodes
    //
    for (std::list<AstNode*>::iterator it = root->children.begin();
         it != root->children.end(); ++it)
    {
        AstNode* node = (*it);

        if (node->type != AST_DECL)
        {
            continue;
        }

        DeclNode* decl = static_cast<DeclNode*>(node);
        
        constantHeaderOutput << "#define " << decl->value.s << " " << translateExpression(decl->expression);
        
        if (decl->lineNumber != 0)
        {
            const char* comment = lookupComment(decl->lineNumber);
            if (comment)
            {
                // Strip the initial ';' character
                //
                constantHeaderOutput << " // " << (comment + 1);
            }
        }

        constantHeaderOutput << "\n";
    }
    
    constantHeaderOutput << "\n" <<
        "#endif // SMBCONSTANTS_HPP\n";
}

void Translator::generateDataDeclarations()
{
    // This is for a structure containing all of the addresses
    // This gets generated as a header file
    //
    std::stringstream addresses;

    addresses <<
        "// Data Addresses (16-bit pointers) for Constants\n" <<
        "//\n" <<
        "struct SMBDataPointers\n" <<
        "{\n";
    
    // This is for a list of defines that accesses the pointers inside
    // the SMBData struct
    //
    std::stringstream addressDefines;

    addressDefines <<
        "// Defines for quick access of the addresses within SMBDataPointers\n" <<
        "//\n\n";
    
    // This is for the default constructor of SMBData, which initializes the
    // pointers in the struct
    std::stringstream addressDefaults;
    addressDefaults << 
        TAB << "SMBDataPointers()\n" <<
        TAB << "{\n";

    // This is for the actual code that does the loading of the data
    // Also, contains the constant data
    //
    std::stringstream loading;

    loading << "void SMBEngine::loadConstantData()\n{\n";

    // Start storing stuff at 0x8000
    //
    int storageAddress = 0x8000;

    for (std::list<AstNode*>::iterator it = root->children.begin();
         it != root->children.end(); ++it)
    {
        AstNode* node = (*it);

        // Skip non-labels
        //
        if (node->type != AST_LABEL)
        {
            continue;
        }

        LabelNode* label = static_cast<LabelNode*>(node);

        // Strip the trailing ':' character
        //
        std::string labelName = label->value.s;
        labelName = labelName.substr(0, labelName.size() - 1);

        if (label->labelType == LABEL_DATA || label->labelType == LABEL_ALIAS)
        {
            if (label->labelType == LABEL_DATA)
            {
                // The constant data declaration
                //
                loading << 
                    TAB << "// " << labelName << "\n" <<
                    TAB << "//\n" <<
                    TAB << "const uint8_t " << labelName << "_data[] = {";
                
                // Translate each data item stored in the label
                //
                ListNode* listElement = static_cast<ListNode*>(label->child);
                assert(listElement);

                int byteCount = 0;
                while (listElement != NULL)
                {
                    loading << "\n" << TAB << TAB;

                    AstNode* dataItem = listElement->value.node;

                    assert(dataItem->type == AST_DATA8);

                    ListNode* dataListElement = static_cast<ListNode*>(dataItem->value.node);
                    assert(dataListElement->type == AST_LIST);
                    while (dataListElement != NULL)
                    {
                        // Translate the data item's expression...
                        //
                        loading << translateExpression(dataListElement->value.node);

                        if (dataListElement->next != NULL)
                        {
                            loading << ", ";
                        }

                        byteCount++;

                        dataListElement = static_cast<ListNode*>(dataListElement->next);
                    }

                    if (listElement->next != NULL)
                    {
                        if (listElement->next->value.node->type != AST_DATA16)
                        {
                            loading << ",";
                        }
                        else
                        {
                            // This only occurs at the very end of the disassembly (the interrupt vectors)
                            // So we will ignore it
                            //
                            break;
                        }
                    }

                    // Add comments at the end of the line (if they exist)
                    //
                    if (dataItem->lineNumber != 0)
                    {
                        const char* comment = lookupComment(dataItem->lineNumber);
                        if (comment)
                        {
                            // Strip the first ';' character
                            //
                            loading << " // " << (comment + 1);
                        }
                    }

                    listElement = static_cast<ListNode*>(listElement->next);
                }
                
                loading << "\n" << TAB << "};\n";

                // The data loading
                //
                loading << TAB << "writeData(" << labelName << ", " << labelName << "_data, sizeof(" << labelName << "_data));\n\n";

                // The addresses that point to the data
                //
                addresses << TAB << "uint16_t " << labelName << "_ptr;\n";

                addressDefines << "#define " << labelName << " (dataPointers." << labelName << "_ptr)\n";

                addressDefaults << TAB << TAB <<  "this->" << labelName << "_ptr = 0x" << std::hex << storageAddress << std::dec << ";\n";

                storageAddress += byteCount;
            }
            // Alias label
            //
            else
            {
                // For aliases, we only need to worry about referencing the aliased value
                // (which will be the next label that we process, hence no changing of the
                //  next storage address)
                //
                addresses << TAB << "uint16_t " << labelName << "_ptr; // alias\n";

                addressDefines << "#define " << labelName << " (dataPointers." << labelName << "_ptr)\n";

                addressDefaults << TAB << TAB <<  "this->" << labelName << "_ptr = 0x" << std::hex << storageAddress << std::dec << ";\n";
            }
        }
    }

    // Final stuff for the end of each section...
    //
    addressDefaults << TAB << "}\n";
    addresses << "\n" << addressDefaults.str();
    addresses << "};\n\n";

    addressDefines << "\n";

    loading << "}\n\n";

    // Put everything in the correct file
    //
    dataHeaderOutput << 
        "#ifndef SMBDATAPOINTERS_HPP\n" <<
        "#define SMBDATAPOINTERS_HPP\n\n";
    dataHeaderOutput << addresses.str();
    dataHeaderOutput << addressDefines.str();
    dataHeaderOutput<< "#endif // SMBDATAPOINTERS_HPP\n";

    sourceOutput << loading.str();
}

void Translator::indexEmptyLines()
{
    std::ifstream file(inputFilename.c_str());

    std::string line;
    int lineNumber = 1;
    while (std::getline(file, line))
    {
        if (line.size() == 0)
        {
            mapNewline(lineNumber);
        }

        lineNumber++;
    }    
}

void Translator::translate()
{
    // Find all empty lines in the file
    //
    indexEmptyLines();

    // Find all data and code labels
    //
    classifyLabels();

    // Begin generating output
    //

    // Put required header
    //
    sourceOutput << "#include \"SMB.hpp\"\n\n";

    // Generate constant declarations first
    //
    generateConstantDeclarations();

    // Generate data declarations next
    //
    generateDataDeclarations();

    // Finally, generate code
    //
    generateCode();
}

std::string Translator::translateBranch(const std::string& condition, const std::string& destination)
{
    std::string result = "if (";
    result += condition;
    result += ")\n";
    result += TAB;
    result += TAB;
    result += "goto ";
    result += destination;
    result += ";";
    
    return result;
}

std::string Translator::translateExpression(AstNode* expr)
{
    std::string result = "";

    if (expr != NULL)
    {
        switch (expr->type)
        {
        case AST_NAME:
            result = expr->value.s;
            break;
        case AST_CONST:
            switch (expr->value.s[0])
            {
            case '$':
                result = "0x";
                result += (expr->value.s + 1);
                break;
            case '%':
                result = "BOOST_BINARY(";
                result += (expr->value.s + 1);
                result += ")";
                break;
            default:
                result = expr->value.s;
                break;
            }
            break;
        case AST_IMMEDIATE:
            // Take the internal value literally
            //
            result = translateExpression(static_cast<UnaryNode*>(expr)->child);
            break;
        case AST_ADD:
            {
                BinaryNode* node = static_cast<BinaryNode*>(expr);
                result = translateExpression(node->lhs) + " + " + translateExpression(node->rhs);
            }
            break;
        case AST_SUBTRACT:
            {
                BinaryNode* node = static_cast<BinaryNode*>(expr);
                result = translateExpression(node->lhs) + " - " + translateExpression(node->rhs);
            }
            break;
        case AST_HIBYTE:
            result = "HIBYTE(";
            result += translateExpression(static_cast<UnaryNode*>(expr)->child);
            result += ")";
            break;
        case AST_LOBYTE:
            result = "LOBYTE(";
            result += translateExpression(static_cast<UnaryNode*>(expr)->child);
            result += ")";
            break;
        case AST_INDIRECT:
            result = "M(";
            result += translateExpression(static_cast<UnaryNode*>(expr)->child);
            result += ")";
            break;
        case AST_INDEXED_X:
            result = translateExpression(static_cast<UnaryNode*>(expr)->child);
            result += " + x";
            break;
        case AST_INDEXED_Y:
            {
                // Check for the special case. If the child expression is indirect,
                // then we must use a 16-bit indirect lookup instead of 8-bit
                //
                AstNode* child = static_cast<UnaryNode*>(expr)->child;
                if (child->type == AST_INDIRECT)
                {
                    result = "W(";
                    result += translateExpression(static_cast<UnaryNode*>(child)->child);
                    result += ") + y";
                }
                else
                {
                    result = translateExpression(child);
                    result += " + y";
                }
            }
            break;
        default:
            assert(false);
            break;                
        }
    }

    return result;
}

std::string Translator::translateInstruction(InstructionNode* inst)
{
    std::string result = "";

    switch (inst->code)
    {
    case LDA:
        result += "a = ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case LDX:
        result += "x = ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case LDY:
        result += "y = ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case STA:
        result += "writeData(";
        result += translateExpression(inst->value.node);
        result += ", a);";
        break;
    case STX:
        result += "writeData(";
        result += translateExpression(inst->value.node);
        result += ", x);";
        break;
    case STY:
        result += "writeData(";
        result += translateExpression(inst->value.node);
        result += ", y);";
        break;
    case TAX:
        result = "x = a;";
        break;
    case TAY:
        result ="y = a;";
        break;
    case TXA:
        result = "a = x;";
        break;
    case TYA:
        result = "a = y;";
        break;
    case TSX:
        result = "x = s;";
        break;
    case TXS:
        result = "s = x;";
        break;
    case PHA:
        result = "pha();";
        break;
    case PHP:
        result = "php();";
        break;
    case PLA:
        result = "pla();";
        break;
    case PLP:
        result = "plp();";
        break;
    case AND:
        result += "a &= ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case EOR:
        result += "a ^= ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case ORA:
        result += "a |= ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case BIT:
        result += "bit(";
        result += translateOperand(inst->value.node);
        result += ");";
        break;
    case ADC:
        result += "a += ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case SBC:
        result += "a -= ";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case CMP:
        result += "compare(a, ";
        result += translateOperand(inst->value.node);
        result += ");";
        break;
    case CPX:
        result += "compare(x, ";
        result += translateOperand(inst->value.node);
        result += ");";
        break;
    case CPY:
        result += "compare(y, ";
        result += translateOperand(inst->value.node);
        result += ");";
        break;
    case INC:
        result += "++";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case INX:
        result += "++x;";
        break;
    case INY:
        result += "++y;";
        break;
    case DEC:
        result += "--";
        result += translateOperand(inst->value.node);
        result += ";";
        break;
    case DEX:
        result += "--x;";
        break;
    case DEY:
        result += "--y;";
        break;
    case ASL:
        if (inst->value.node)
        {
            result += translateOperand(inst->value.node);
            result += " <<= 1;";
        }
        else
        {
            result += "a <<= 1;";
        }
        break;
    case LSR:
        if (inst->value.node)
        {
            result += translateOperand(inst->value.node);
            result += " >>= 1;";
        }
        else
        {
            result += "a >>= 1;";
        }
        break;
    case ROL:
        if (inst->value.node)
        {
            result += translateOperand(inst->value.node);
            result += ".rol();";
        }
        else
        {
            result += "a.rol();";
        }
        break;
    case ROR:
        if (inst->value.node)
        {
            result += translateOperand(inst->value.node);
            result += ".ror();";
        }
        else
        {
            result += "a.ror();";
        }
        break;
    case JMP:
        {
            // We only care about jumping to labels
            // Jumping to a referenced address is only used once in JumpEngine,
            // which we reimplement in a different way
            //
            if (inst->value.node->type == AST_NAME)
            {
                // Check for special case (jmp EndlessLoop)
                //
                if (strcmp(inst->value.node->value.s, "EndlessLoop") == 0)
                {
                    result += "return;";
                }
                else
                {
                    result += "goto ";
                    result += translateExpression(inst->value.node);
                    result += ";";
                }
            }
        }
        break;
    case JSR:
        {
            // Check for special case (jsr JumpEngine)
            //
            if (strcmp(inst->value.s, "JumpEngine") == 0)
            {
                // Create a switch-case jump table
                // using the labels that follow as data
                //
                ListNode* listElement = static_cast<ListNode*>(inst->parent);
                assert(listElement != NULL);

                result += "switch (a)\n";
                result += TAB;
                result += "{\n";

                // Skip our element
                //
                listElement = static_cast<ListNode*>(listElement->next);
                int index = 0;
                while (listElement != NULL)
                {
                    result += TAB;
                    result += "case ";

                    char indexStr[8];
                    sprintf(indexStr, "%d", index);
                    result += indexStr;
                    result += ":\n";

                    // A little funky...
                    // We have an outer list element,
                    // which contains a 16-bit data list element (.dw),
                    // which contains another list element,
                    // which contains the name that we want to add to our jump table
                    //
                    result += TAB;
                    result += TAB;
                    result += "goto ";
                    result += translateExpression(listElement->value.node->value.node->value.node);
                    result += ";";

                    // Add comments at the end of the line (if they exist)
                    //
                    if (listElement->value.node->lineNumber != 0)
                    {
                        const char* comment = lookupComment(listElement->value.node->lineNumber);
                        if (comment)
                        {
                            // Strip the first ';' character
                            //
                            result += " // ";
                            result += (comment + 1);
                        }
                    }                    

                    result += "\n";

                    listElement = static_cast<ListNode*>(listElement->next);
                    index++;
                }
                
                result += TAB; 
                result += "}";
            }
            else
            {
                char returnLabelIndexStr[8];
                sprintf(returnLabelIndexStr, "%d", returnLabelIndex++);
                result += "JSR(";
                result += inst->value.s;
                result += ", ";
                result += returnLabelIndexStr;
                result += ");";
            }
        }
        break;
    case RTS:
        result += "goto Return;";
        break;
    case BCC:
        result = translateBranch("!c", inst->value.s);
        break;
    case BCS:
        result = translateBranch("c", inst->value.s);
        break;
    case BEQ:
        result = translateBranch("z", inst->value.s);
        break;
    case BMI:
        result = translateBranch("n", inst->value.s);
        break;
    case BNE:
        result = translateBranch("!z", inst->value.s);
        break;
    case BPL:
        result = translateBranch("!n", inst->value.s);
        break;
    case BVC:
        // NYI
        //
        assert(false);
        break;
    case BVS:
        // NYI
        //
        assert(false);
        break;
    case CLC:
        result = "c = 0;";
        break;
    case CLD:
        // IGNORE
        //
        result = "/* cld */";
        break;
    case CLI:
        // NYI
        //
        assert(false);
        break;
    case CLV:
        // NYI
        //
        assert(false);
        break;
    case SEC:
        result = "c = 1;";
        break;
    case SED:
        // NYI
        //
        assert(false);
        break;
    case SEI:
        // IGNORE
        //
        result = "/* sei */";
        break;
    case BRK:
        // NYI
        //
        assert(false);
        break;
    case NOP:
        result += "; // nop";
        break;
    case RTI:
        result += "return;";
        break;
    default:
        assert(false);
        break;
    }

    return result;
}

std::string Translator::translateOperand(AstNode* operand)
{
    std::string result = "";

    if (operand != NULL)
    {
        // If immediate addressing is not the underlying node, it means read from memory
        //
        if (operand->type != AST_IMMEDIATE)
        {
            result = "M(";
            result += translateExpression(operand);
            result += ")";
        }
        else
        {
            result = translateExpression(operand);
        }
    }

    return result;
}
