//===- mlirvisitor.h ------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
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
//
// This file is about buddy-toy-dsl how to generate mlir in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_VISITOR_H
#define MLIR_VISITOR_H

#include "ToyBaseVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <numeric>
#include <stdlib.h>
#include <vector>

class MLIRVisitor : public ToyBaseVisitor {
  // symbolTable to store variables.
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  // builder to create op.
  mlir::OpBuilder builder;
  // check whether we should return automaticly.
  bool returnFlag = false;
  // use filename to contain the name of file.
  std::string fileName;
  // use vecDecl to store shared_ ptr.
  std::vector<std::shared_ptr<std::string>> vecDecl;

  // declare variable.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  // we input line and row of a symbol.
  // and return the type mlir::Location variable to create op.
  mlir::Location loc(int line, int row) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(fileName), line,
                                     row);
  }

  // convert dimension to mlir type.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  // visitFundefinition can process define function.
  virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) override {
    returnFlag = false;
    // for every function we should have a symbolTable to store the variables.
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);
    builder.setInsertionPointToEnd(theModule.getBody());
    // process function type.
    visit(ctx->prototype());
    // process function block.
    visit(ctx->block());
    vecDecl.clear();
    // if input code no return expression,we generate return automaticly.
    if (!returnFlag) {
      auto location =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      builder.create<mlir::toy::ReturnOp>(location,
                                          llvm::ArrayRef<mlir::Value>());
    }
    return 0;
  }

  // visitPrototype can process the type of function.
  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) override {
    mlir::Location location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    auto varNumber = 0;
    // get the number of arguments.
    if (ctx->declList()) {
      auto list = ctx->declList();
      while (list->Identifier()) {
        varNumber++;
        if (list->declList())
          list = list->declList();
        else
          break;
      }
    }

    llvm::SmallVector<mlir::Type, 4> argTypes(
        varNumber, mlir::UnrankedTensorType::get(builder.getF64Type()));
    auto funType = builder.getFunctionType(argTypes, llvm::None);
    auto func = builder.create<mlir::toy::FuncOp>(
        location, ctx->Identifier()->toString(), funType);
    mlir::Block &entryblock = func.front();
    builder.setInsertionPointToStart(&entryblock);
    return 0;
  }

  // the visitExpression can process expression.
  virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) override {
    mlir::Value value;
    // if the expression is tensorLiteral.
    if (ctx->tensorLiteral()) {
      return tensor(ctx->tensorLiteral());
    } else if (ctx->identifierExpr()) { // if the expression is call function or
                                        // variable.
      return visit(ctx->identifierExpr());
    }
    return value;
  }
  // the tensor can only process two dimension tensor.
  std::any tensor(ToyParser::TensorLiteralContext *ctx) {
    bool flag = false;
    std::vector<int64_t> dims;
    std::vector<double> data;
    // get dimensions.
    dims.push_back(ctx->Comma().size() + 1);
    if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
      flag = true;
      dims.push_back(ctx->tensorLiteral(0)->Comma().size() + 1);
    }
    // get data of tensorLiteral.
    auto list = ctx;
    if (flag)
      for (auto i : ctx->tensorLiteral()) {
        for (auto j : i->tensorLiteral()) {
          data.push_back(std::atof(j->Number()->toString().c_str()));
        }
      }
    else if (!flag) {
      for (auto i : ctx->tensorLiteral()) {
        data.push_back(std::atof(i->Number()->toString().c_str()));
      }
    }
    mlir::Type elementType = builder.getF64Type();
    auto type = getType(dims);
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));
    auto loaction =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value =
        builder.create<mlir::toy::ConstantOp>(loaction, type, dataAttribute);
    return value;
  }

  // visitDecl can process define a variable.
  virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) override {
    mlir::Value value = std::any_cast<mlir::Value>(visit(ctx->expression()));
    // if the variable has shape, we should create ReshapeOp.
    if (ctx->type()) {
      std::vector<int64_t> v0;
      auto v1 = ctx->type()->Number();
      for (auto i : v1) {
        auto j = atoi(i->toString().c_str());
        v0.push_back(j);
      }
      mlir::Location location =
          loc(ctx->Identifier()->getSymbol()->getLine(),
              ctx->Identifier()->getSymbol()->getCharPositionInLine());
      value =
          builder.create<mlir::toy::ReshapeOp>(location, getType(v0), value);
    }
    auto var = std::make_shared<std::string>(ctx->Identifier()->toString());
    vecDecl.push_back(var);
    // store the variable in the symboltable.
    mlir::failed(declare(*var, value));
    return 0;
  }
  // visitIdentifierexpr can process call function and variables.
  virtual std::any
  visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) override {
    mlir::Value value;
    // this is call function.
    if (ctx->ParentheseOpen()) {
      auto location =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      llvm::SmallVector<mlir::Value, 4> oprands;
      int num = 0;

      for (auto i : ctx->expression()) {
        mlir::Value arg = std::any_cast<mlir::Value>(visit(i));
        oprands.push_back(arg);
      }
      // if the name of function is print, we create PrintOp.
      if (ctx->Identifier()->toString() == "print") {
        auto arg = oprands[0];
        builder.create<mlir::toy::PrintOp>(location, arg);
        return 0;
      }
      // if the name of funcion is not print, we create GenericCallOp.
      value = builder.create<mlir::toy::GenericCallOp>(
          location, ctx->Identifier()->toString(), oprands);
      return value;
    } else { // process variable.
      value = symbolTable.lookup(ctx->Identifier()->toString());
      return value;
    }
  }

  // visitReturnExpression can process the return expression.
  virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) override {
    returnFlag = true;
    auto location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value expr = nullptr;
    if (ctx->expression()) {
      expr = std::any_cast<mlir::Value>(ctx->expression());
    }
    builder.create<mlir::toy::ReturnOp>(location,
                                        expr ? llvm::makeArrayRef(expr)
                                             : llvm::ArrayRef<mlir::Value>());
    return 0;
  }

public:
  MLIRVisitor(std::string filename) : fileName(filename), builder(&context) {
    context.getOrLoadDialect<mlir::toy::ToyDialect>();
    builder = mlir::OpBuilder(&context);
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  mlir::ModuleOp theModule;
  // context store the Dialect we define.
  mlir::MLIRContext context;
};

#endif
