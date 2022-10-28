# FrontendGen

因为做编译器前端是一个繁琐枯燥的过程，所以我们想让写前端的人更加轻松，基于这个想法，FrontendGen诞生了，通过写语法生成式，以及少量的标记，可以生成ast visitor 和 Dialect的一部分映射，不过目前FrontendGen能做的事情其实还很有限,所以FrontendGen还只是一个demo。

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

  我们用上面那的代码代码作为输入文件。**rule** 用来说明 expression是非终结符，后面可以写多条生成式。在这里包括了**expression 'mul' expression**

  **expression Add expression**。这里定义了一个mul的终结符，终结符的定义的方法是把单词用单引号引起来。

  如果想要生成g4文件，可以输入以下的参数。

  ```bash
  ./buddy-frontendgen -f test.fegen -emit=antlr -g Toy
  ```

  **-f** 参数用来指定是哪个文件。

  **-emit** 用来指定要生成的是什么类型的文件。

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

  输入上面的命令会得到上面的文件，你能够看到**Mul**, 这里**Mul**是我们刚刚定义的终结符，**grammar Toy**是文法的名称，后面包含了一系列FrontendGen中内置的终结符。

 * -emit=ast

   如果生成的文件错误，你可以通过使用**-emit=ast**进行调试。用上面的输入文件作为一个例子。

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

   **rule name** 用来说明这是哪一条规则，而下面的的**generator**是这条规则下面的生成式，**(rule)** 用来说明**"expression"**是非终结符，而**(terminator)**说明**Mul** 和 **Add**是终结符。我们可以通过生成的ast进程调试，查看代码是否出现问题。

* 正则表达式

  我们也支持正则表达式

  ```c
  rule expression
    : expression 'Mul'+ expression
    : expression Add? expression
    ;
  ```

  查看输出的文件

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

  **bpExpression**用来说明**"+"**和**"?"**是正则表达式。

  
