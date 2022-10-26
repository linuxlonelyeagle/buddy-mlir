//====- buddy-opt.cpp - The driver of buddy-mlir --------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "CGModule.h"
#include "llvm/Support/Casting.h"

using namespace frontendgen;

/// Emit the ast,currently only antlr's ast are supported.
void CGModule::emitAST() {
  for (auto i : module->getRules()) {
    llvm::outs() << "rule name: " << i->getName() << '\n';
    for (auto j : i->getGeneratorsAndOthers()) {
      llvm::outs() << "  generator: " << '\n' << "    ";
      for (auto k : j->getGenerator()) {
        if (k->getKind() == AntlrBase::baseKind::rule)
          llvm::outs() << "\"" << k->getName() << "\"(rule) ";
        else if (k->getKind() == AntlrBase::baseKind::terminator)
          llvm::outs() << "\"" << k->getName() << "\"(terminator) ";
        else if (k->getKind() == AntlrBase::baseKind::pbexpression)
          llvm::outs() << "\"" << k->getName() << "\"(bpExpression) ";
      }
      llvm::outs() << '\n';
    }
  }
}

/// Emit the code of antlr , emit the generative formula first, then emit
/// user-defined terminator , and finally emit the system-defined terminator.
void CGModule::emitAntlr(llvm::StringRef grammarName) {
  emitGrammar(grammarName);
  emit(module->getRules());
  emitCustomTerminators();
  emitTerminators();
}
/// Emit the system-defined terminator.
void CGModule::emitTerminators() {
  for (auto start = terminators.terminators.begin();
       start != terminators.terminators.end(); start++) {
    os << start->first() << '\n';
    os << "  : " << start->second << "\n  ;\n\n";
  }
  emitWSAndComment();
}

void CGModule::emitGrammar(llvm::StringRef grammarNmae) {
  os << "grammar " << grammarNmae << ";\n\n";
}

/// Emit user-defined terminator.
void CGModule::emitCustomTerminators() {
  for (auto terminator : terminators.customTerminators) {
    std::string tmp = terminator.str();
    if (tmp[0] >= 'a' && tmp[0] <= 'z')
      tmp[0] -= 32;
    llvm::StringRef name(tmp);
    os << name << '\n';
    os << "  : \'" << terminator.str() << "\'\n  ;\n\n";
  }
}

/// Emit the generative formula.
void CGModule::emit(const std::vector<Rule *> &rules) {
  for (Rule *rule : rules) {
    os << rule->getName() << '\n';
    emit(rule->getGeneratorsAndOthers());
    os << '\n';
  }
}

/// Emit all generative formuls  in a rule.
void CGModule::emit(
    const std::vector<GeneratorAndOthers *> &generatorsAndOthers) {
  for (GeneratorAndOthers *generatorAndOthers : generatorsAndOthers) {
    if (generatorAndOthers == generatorsAndOthers[0])
      os << "  : ";
    else
      os << "  | ";
    emit(generatorAndOthers->getGenerator());
  }
  os << "  ;\n";
}

/// Output the elements of the generated formula.
void CGModule::emit(const std::vector<AntlrBase *> &generator) {
  for (AntlrBase *base : generator) {
    if (base->getKind() == AntlrBase::baseKind::terminator) {
      std::string tmp = base->getName().str();
      // The terminator in antlr must be capitalized.
      if (tmp[0] >= 'a' && tmp[0] <= 'z')
        tmp[0] -= 32;
      llvm::StringRef name(tmp);
      os << name << " ";
    } else if (base->getKind() == AntlrBase::baseKind::rule) {
      os << base->getName() << " ";
    } else if (base->getKind() == AntlrBase::baseKind::pbexpression) {
      os << base->getName();
    }
  }
  os << '\n';
}

/// TODO: Supports user-defined comment whitespace.
void CGModule::emitWSAndComment() {
  os << "Identifier\n  : [a-zA-Z][a-zA-Z0-9_]*\n  ;\n\n";
  os << "WS\n  : [ \\r\\n\\t] -> skip\n  ;\n\n";
  os << "Comment\n  : '#' .*? \'\\n\' ->skip\n  ;\n";
}

void CGModule::emitMLIRVisitor(llvm::StringRef grammarName) {
  emitIncludes(grammarName);
  emitClass(grammarName);
}

void CGModule::emitIncludes(llvm::StringRef grammarName) {
  os << "#include \"" << grammarName << "BaseVisitor.h\"\n";
  os << "#include \"" << grammarName << "Lexer.h\"\n";
  os << "#include \"" << grammarName << "Parser.h\"\n";
  os << "#include \"mlir/IR/Attributes.h\"\n";
  os << "#include \"mlir/IR/Builders.h\"\n";
  os << "#include \"mlir/IR/BuiltinOps.h\"\n";
  os << "#include \"mlir/IR/BuiltinTypes.h\"\n";
  os << "#include \"mlir/IR/MLIRContext.h\"\n";
  os << "#include \"mlir/IR/Verifier.h\"\n";
  os << "#include \"llvm/ADT/STLExtras.h\"\n";
  os << "#include \"llvm/ADT/ScopedHashTable.h\"\n";
  os << "#include \"llvm/ADT/StringRef.h\"\n";
  os << "#include \"llvm/Support/raw_ostream.h\"\n";
  os << "\n";
}

/// Emit visitor class.
void CGModule::emitClass(llvm::StringRef gramarName) {
  os << "class MLIR" << gramarName << "Visitor : public " << gramarName
     << "BaseVisitor {\n";

  os << "mlir::ModuleOp theModule;\n";
  os << "mlir::OpBuilder builder;\n";
  os << "std::string fileName;\n\n";

  os << "public:\n";
  os << "MLIR" << gramarName
     << "Visitor(std::string filename, mlir::MLIRContext &context)\n"
     << ": builder(&context), fileName(filename) "
     << "{\n theModule = mlir::ModuleOp::create(builder.getUnknownLoc()); "
        "\n}\n\n";
  os << "mlir::ModuleOp getModule() { return theModule; }\n\n";

  // Emit all virtual functions.
  auto rules = module->getRules();
  for (auto rule : rules) {
    emitRuleVisitor(gramarName, rule);
  }
  os << "};\n";
}
/// Emit virtual function in antlr.
void CGModule::emitRuleVisitor(llvm::StringRef grammarName, Rule *rule) {
  std::string ruleName = rule->getName().str();
  ruleName[0] = ruleName[0] - 32;
  os << "virtual std::any visit" << ruleName;
  os << "(" << grammarName << "Parser::" << ruleName << "Context *ctx) {\n";
  emitBuilders(rule);
  os << "  return visitChildren(ctx);\n";
  os << "}\n\n";
}

void CGModule::emitBuilders(Rule *rule) {
  for (GeneratorAndOthers *generatorAndOthers :
       rule->getGeneratorsAndOthers()) {
    llvm::SmallVector<llvm::StringRef> builderOpNames =
        generatorAndOthers->getBuilderNames();
    llvm::SmallVector<int> indexs = generatorAndOthers->getbulderIdxs();
    int size = builderOpNames.size();
    for (int start = 0; start < size; start++)
      emitBuilder(builderOpNames[start], indexs[start]);
  }
}

void CGModule::emitBuilder(llvm::StringRef builderOp, int index) {
  Op *op = findOp(builderOp);
  if (op == nullptr) {
    llvm::errs() << builderOp << " is undefined!\n";
    return;
  }
  emitOp(op, index);
}

Op *CGModule::findOp(llvm::StringRef opName) {
  for (Op *op : module->getOps()) {
    if (op->getOpName() == opName)
      return op;
  }
  return nullptr;
}

/// Emit the operation we want to create.
void CGModule::emitOp(Op *op, int index) {
  // Emit the default builder function.
  if (index == 0) {
    DAG *arguments = op->getArguments();
    DAG *result = op->getResults();
    llvm::SmallVector<llvm::StringRef> argOperands;
    llvm::SmallVector<llvm::StringRef> argOperandNames;
    llvm::SmallVector<llvm::StringRef> resOperands;
    llvm::SmallVector<llvm::StringRef> resOperandNames;
    if (arguments) {
      argOperands = arguments->getOperands();
      argOperandNames = arguments->getOperandNames();
    }
    if (result) {
      resOperands = result->getOperands();
      resOperandNames = result->getOperandNames();
    }
    os << "  {\n";
    // opArgments are used to store the names of the arguments needed to create
    // the operation.
    llvm::SmallVector<llvm::StringRef> opArgments;
    // Emit variables for creation operation.
    // Emit variables of result type.
    for (size_t index = 0; index < resOperands.size(); index++) {
      if (!typeMap.findResultsMap(resOperands[index]).empty()) {
        os << "  " << typeMap.findResultsMap(resOperands[index]) << " ";
        if (!resOperandNames[index].empty()) {
          os << resOperandNames[index] << ";\n";
          opArgments.push_back(resOperandNames[index]);
        } else {
          os << "res" << index << ";\n";
          llvm::StringRef arg("res" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else if (resOperands[index].startswith("AnyTypeOf")) {
        llvm::StringRef operand = resOperands[index];
        int start = operand.find('[') + 1;
        int end = operand.find(']');
        int cur = start;
        if (start == std::string::npos || end == std::string::npos) {
          return;
        }
        llvm::StringRef type;
        while (cur <= end) {
          if (operand[cur] == ',' || cur == end) {
            std::string str(operand, start, cur - start);
            str.erase(0, str.find_first_not_of(" "));
            str.erase(str.find_last_not_of(" ") + 1);
            if (typeMap.findResultsMap(str).empty()) {
              llvm::errs() << str << " in " << op->getOpName()
                           << " in results is not supported.\n";
            }
            type = typeMap.findResultsMap(str);
            start = cur + 1;
          }
          cur++;
        }
        os << "  " << type << " ";
        if (!resOperandNames[index].empty()) {
          os << resOperandNames[index] << ";\n";
          opArgments.push_back(resOperandNames[index]);
        } else {
          os << "res" << index << ";\n";
          llvm::StringRef arg("res" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else {
        llvm::errs() << resOperands[index] << " in " << op->getOpName()
                     << " in results is not supported.\n";
        return;
      }
    }
    // Emit variables of argument type.
    for (size_t index = 0; index < argOperands.size(); index++) {
      if (!typeMap.findArgumentMap(argOperands[index]).empty()) {
        os << "  " << typeMap.findArgumentMap(argOperands[index]) << " ";
        if (!argOperandNames[index].empty()) {
          os << argOperandNames[index] << ";\n";
          opArgments.push_back(argOperandNames[index]);
        } else {
          os << "arg" << index << ";\n";
          llvm::StringRef arg("arg" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else if (argOperands[index].startswith("AnyTypeOf")) {
        llvm::StringRef operand = argOperands[index];
        int start = operand.find('[') + 1;
        int end = operand.find(']');
        int cur = start;
        if (start == std::string::npos || end == std::string::npos) {
          return;
        }
        llvm::StringRef type;
        while (cur <= end) {
          if (operand[cur] == ',' || cur == end) {
            std::string str(operand, start, cur - start);
            str.erase(0, str.find_first_not_of(" "));
            str.erase(str.find_last_not_of(" ") + 1);
            if (typeMap.findArgumentMap(str).empty()) {
              llvm::errs() << str << " in " << op->getOpName()
                           << " in arguments is not supported.\n";
            }
            start = cur + 1;
            type = typeMap.findArgumentMap(str);
          }
          cur++;
        }
        os << "  " << type << " ";
        if (!argOperandNames[index].empty()) {
          os << argOperandNames[index] << ";\n";
          opArgments.push_back(argOperandNames[index]);
        } else {
          os << "arg" << index << ";\n";
          llvm::StringRef arg("arg" + std::to_string(index));
          opArgments.push_back(arg);
        }
      } else {
        llvm::errs() << argOperands[index] << " in " << op->getOpName()
                     << " in arguments is not supported.\n";
        return;
      }
    }
    // Emit the operation we want to create.
    os << "  mlir::Location location;\n";
    llvm::StringRef cppNameSpace(
        module->getDialect()->getCppNamespace().data() + 1,
        module->getDialect()->getCppNamespace().size() - 2);
    os << "  "
       << "builder.create<" << cppNameSpace << "::" << op->getOpName()
       << ">(location";
    if (opArgments.size())
      os << ", ";
    for (size_t index = 0; index < opArgments.size(); index++) {
      os << opArgments[index];
      if (index + 1 != opArgments.size())
        os << ", ";
    }
    os << ");\n";
    os << "  }\n\n";
  } else if (index > 0) {
    // Emit custom builder function.
    index--;
    // Emit the variables which are used to fill builder function.
    llvm::SmallVector<llvm::StringRef, 4> operands =
        op->getBuilders()[index]->getDag()->getOperands();
    llvm::SmallVector<llvm::StringRef, 4> operandNames =
        op->getBuilders()[index]->getDag()->getOperandNames();
    llvm::SmallVector<llvm::StringRef> opArguments;
    os << "  {\n";
    for (size_t index = 0; index < operands.size(); index++) {
      if (!typeMap.findCppMap(operands[index]).empty())
        os << "  " << typeMap.findCppMap(operands[index]);
      else
        os << "  " << operands[index];
      if (!operandNames[index].empty()) {
        os << " " << operandNames[index] << ";\n";
        opArguments.push_back(operandNames[index]);
      } else {
        os << " arg" << index << ";\n";
        opArguments.push_back("arg" + std::to_string(index));
      }
    }
    // Emit the operation we want to create.
    os << "  mlir::Location location;\n";
    llvm::StringRef cppNameSpace(
        module->getDialect()->getCppNamespace().data() + 1,
        module->getDialect()->getCppNamespace().size() - 2);
    os << "  "
       << "builder.create<" << cppNameSpace << "::" << op->getOpName()
       << ">(location";
    if (!operandNames.empty()) {
      os << ", ";
      for (size_t index = 0; index < opArguments.size(); index++) {
        os << opArguments[index];
        if (index + 1 != opArguments.size())
          os << ", ";
      }
    }
    os << ");\n";
    os << "  }\n\n";
  }
}

llvm::StringRef TypeMap::findCppMap(llvm::StringRef key) {
  if (cppMap.find(key) == cppMap.end())
    return llvm::StringRef();
  return cppMap[key];
}

llvm::StringRef TypeMap::findArgumentMap(llvm::StringRef key) {
  if (argmentsMap.find(key) == argmentsMap.end())
    return llvm::StringRef();
  return argmentsMap[key];
}

llvm::StringRef TypeMap::findResultsMap(llvm::StringRef key) {
  if (resultsMap.find(key) == resultsMap.end())
    return llvm::StringRef();
  return resultsMap[key];
}
