# FrontendGen

做编译器前端是一个繁琐枯燥的过程，我们的初衷是想让实现dsl更加容易，让写前端的程序员更加轻松，基于这个想法，FrontendGen诞生了，通过写语法生成式，以及少量的标记，可以生成ast visitor 和 Dialect的一部分映射，不过目前FrontendGen能做的事情其实还很有限,FrontendGen还只是一个demo。

## 使用教程

### 生成g4文件

FrontendGen可以生成g4文件，g4文件再交给antlr进行处理，也就是说我们采用antlr作为后端，g4文件作为中间表示，如果生成的g4文件不是很满意，也可以在g4文件上做出一些修改。

* 内置的终结符

  ```c
  Var 
      : 'var'
      ;
  
  Add 
      : 'add'
      ;
  
  Sub 
      : 'sub'
      ;
  
  Def 
      : 'def'
      ;
  
  Return 
      : 'return'
      ;
  
  ParentheseOpen 
      : '('
      ;
  
  ParentheseClose 
      : ')'
      ;
  
  Comma 
      : ','
      ;
  
  BracketOpen 
      : '{'
      ;
  
  BracketClose 
      : '}'
      ;
  
  SbracketOpen 
      :  '['
      ;
  
  SbracketClose 
      : ']'
      ;
  
  Semi 
      : ';'
      ;
  
  AngleBracketOpen 
      : '<'
      ;
  
  AngleBracketClose 
      : '>'
      ;
  
  Number 
      : [0-9]
  	;
  
  Equal 
      : '='
      ;
  
  Identifier 
      : [a-zA-Z][a-zA-Z0-9_]*
      ;
  
  WS
      : [ \r\n\t] -> skip
      ;
  
  Comment 
      : '#' .*? '\n' ->skip
      ;
  ```
  
  我们为用户提供了一些比较常见的终结符，这些终结符会自动包含在g4文件中，用户不需要自己生成这些终结符,当用户需要使用一些终结符的时候，可以随时查阅这张表。
  
* 自定义终结符

  ```c
  rule expression
    : expression 'mul' expression
    : expression Add expression
    ;
  ```

  我们用上面那的代码代码作为输入文件。rule 关键字用来说明expression是非终结符，后面可以有多条生成式。在这里有expression 'mul' expression

  expression Add expression。这里定义了一个mul的终结符，终结符的定义的方法是把标识符用单引号引起来。

  如果想要生成g4文件，可以输入以下的命令。

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=antlr -g Toy
  ```

  **-f** 参数用来指定输入文件。

  **-emit** 用来指定生成文件的类型。

  **-g** 用来指定文法的名称，也用来指定生成文件的名称。

  ```c
  grammar Toy;
  
  expression
    : expression Mul expression 
    | expression Add expression 
    ;
  
  Mul
    : 'mul'
    ;
  ```

  输入上面的命令会得到Toy.g4文件，你能够看到Mul, 这里Mul是我们刚刚定义的终结符，grammar Toy是文法的名称，文件后面还包含了一系列FrontendGen中内置的终结符，我没有列出来。

 * -emit=ast

   如果生成的文件内容有错误，你可以通过使用-emit=ast进行调试，用上面的输入文件作为一个例子。

   ```bash
   ./buddy-frontendgen -f test.fegen -emit=ast
   ```

   你会得到下面的结果

   ```bash
   rule name: expression
     generator: 
       "expression"(rule) "Mul"(terminator) "expression"(rule) 
     generator: 
       "expression"(rule) "Add"(terminator) "expression"(rule) 
   ```

   rule name 用来说明这是哪一条规则，而下面的的generator是这条规则下面的生成式，rule用来说明expression是非终结符，而terminator说明Mul 和Add是终结符。

* 正则表达式

  我们也支持正则表达式

  ```c
  rule expression
    : expression 'Mul'+ expression
    : expression Add? expression
    ;
  ```

  输入下面的命令

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=antlr -g Toy
  ```

  ```c
  grammar Toy;
  
  expression
    : expression Mul +expression 
    | expression Add ?expression 
    ;
  
  Mul
    : 'Mul'
    ;
  ```

  查看ast

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=ast
  ```

  ```c
  rule name: expression
    generator: 
      "expression"(rule) "Mul"(terminator) "+"(bpExpression) "expression"(rule) 
    generator: 
      "expression"(rule) "Add"(terminator) "?"(bpExpression) "expression"(rule) 
  ```

  bpExpression用来说明+和?是正则表达式。

### 生成MLIRvisitor文件

* 生成简单的visitor

  以上面的代码作为例子

  ```c
  rule expression
    : expression 'mul' expression
    : expression Add expression
    ;
  ```

  输入下面的命令

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=visitor -g Toy
  ```

  -emit=visitor用来生成visitor文件，输入上面的命令，我们会得到一个MLIRToyVisitor.h的文件，下面是这个文件的内容。在这里，我们帮助用户定义了MLIRxxxVisitor类，xxx和生成的文件的名称与指定的文法的名称有关，因为我们指定了-g Toy，Toy是文法的名称。

  ```c++
  #include "mlir/IR/Builders.h"
  #include "mlir/IR/BuiltinOps.h"
  #include "mlir/IR/BuiltinTypes.h"
  #include "mlir/IR/MLIRContext.h"
  #include "mlir/IR/Verifier.h"
  #include "llvm/ADT/STLExtras.h"
  #include "llvm/ADT/ScopedHashTable.h"
  #include "llvm/ADT/StringRef.h"
  #include "llvm/Support/raw_ostream.h"
  
  class MLIRToyVisitor : public ToyBaseVisitor {
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  std::string fileName;
  
  public:
  MLIRToyVisitor(std::string filename, mlir::MLIRContext &context)
  : builder(&context), fileName(filename) {
   theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  
  mlir::ModuleOp getModule() { return theModule; }
  
  virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) {
    return visitChildren(ctx);
  }
  };
  ```

  在生成的类里面，我们帮助用户生成了一些比较实用的头文件，还有一些方法，帮助用户自动写出来有那些虚函数，用户可以在里面写具体的实现，虚函数有哪些是由test.fegen中的rule指示的，也就是说每一条规则都会生成一个虚函数。

* 在visitor文件生成构造Op的函数

  我们为用户提供了自动构造Op的函数的功能，我们用下面的代码作为例子,写入test.fegen文件中。

  ```c
  dialect Toy_Dialect
    : name = "toy"
    : cppNamespace = "mlir::toy"
    ;
  
  
  op FuncOp
    : arguments = (ins
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type
    )
    : builders = [ OpBuilder<(ins
      "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ]
    ;
  
  rule prototype
    : Def Identifier ParentheseOpen declList ? ParentheseClose {
      builder = FuncOp_0
    } 
    ;
  
  rule declList
    : Identifier
    : Identifier Comma declList
    ;
  ```

  首先我们定义了一个Toy_Dialect,name是这个dialect的名称，cppNamespace用来指定Toy_Dialect在c++中的命名空间，下面的Op默认也会在这个命名空间中。然后用op关键字定义了FuncOp，里面的arguments和builders全部都是能够用来生成FuncOp的描述。随后在prototype规则中添加了一行代码，如果想自动生成Op，必须要用{}把下面的代码包起来。

  ```c
  builder = FuncOp_0
  ```

  输入下面的命令

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=visitor -g Toy
  ```

  ```c++
  
  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) {
    {
    llvm::StringRef sym_name;
    mlir::FunctionType function_type;
    mlir::Location location;
    builder.create<mlir::toy::FuncOp>(location, sym_name, function_type);
    }
  
    return visitChildren(ctx);
  }
  
  virtual std::any visitDeclList(ToyParser::DeclListContext *ctx) {
    return visitChildren(ctx);
  }
  
  };
  ```

  mlirvisitor.h文件中多余的部分我没有再列出来，我们可以看到在visitPrototype函数中生成了构造FuncOp的函数，值得注意的一点是,FuncOp_0在这里的意思是使用arguments的那条描述生成的构造Op的函数。接下来我们会使用builders中描述的方法来生成构造Op的函数，我们进行如下改动，同时使用了arguments和builders属性中的描述来生成构造FuncOp的函数。

  ```c
  rule prototype
    : Def Identifier ParentheseOpen declList ? ParentheseClose {
      builder = FuncOp_0,FuncOp_1
    } 
    ;
  ```

  我们会得到下面的c++代码

  ```c++
  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) {
    {
    llvm::StringRef sym_name;
    mlir::FunctionType function_type;
    mlir::Location location;
    builder.create<mlir::toy::FuncOp>(location, sym_name, function_type);
    }
  
    {
    llvm::StringRef name;
    llvm::FunctionType type;
    llvm::ArrayRef<NamedAttribute> attrs;
    mlir::Location location;
    builder.create<mlir::toy::FuncOp>(location, name, type, attrs);
    }
  
    return visitChildren(ctx);
  }
  
  virtual std::any visitDeclList(ToyParser::DeclListContext *ctx) {
    return visitChildren(ctx);
  }
  
  };
  ```

  可以看到，visitorPrototype函数中又多了一个构造FuncOp的函数。关于FuncOp_index中index的意义，index用来指示用哪一条描述来生成FuncOp,如果index为0说明使用arguments的描述生成构造FuncOp的函数,如果index大于0,就说明应该用builders中的描述，其中builders中也可以有多条描述生成FuncOp，举个例子，如果index等于1,说明用builders中的第一条描述，如果index等于2,就说明用builders中的第二条描述。
  
  关于生成mlirvisitor的教程其实到这里就结束了，下面进行一点小小的补充。
  
  你可以使用-emit=all同时生成g4文件和mlirvisitor文件，你也可以使用-h或--help参数查看一些信息。

## 最后

FrontendGen最开始只是一个想法，目前还只能作为一个demo，能做的事还是很有限，如果在使用的过程中你有好的想法，欢迎联系我们，如果你发现了bug，欢迎给我们提交issue，最后希望FrontendGen越来越好。
