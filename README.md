# Project Name
A comparison of self-healing parsers and machine learning for repairing syntax error

# Progress

## On Going
The dataset of the paper - [A Large-Scale Empirical Study of Compiler Errors in Continuous Integration](https://compilererrorinci.github.io) might not be useful to this project. Because syntax errors lead to compilation errors but the converse is not always true. It makes me question that **are these papers focusing on syntax errors?**

So I studied how the two papers choose source files and decide whether repair is successful.
* [LR parser-based](https://github.com/softdevteam/error_recovery_experiment): use a java parser (Java 7) to parse the source file. In other words, static analysis. This can be proved in their Github repository.
* [Sensibility](https://github.com/naturalness/sensibility): use `javac` to compile a source file. In other words, runtime analysis. They do not provide code to construct dataset, but they mentioned it in paper (Section VI, A.)


```Java
import java.utillll.*;
public class Giant
{
    // instance variables - replace the example below with your own
    private int x;
    public Giant(){
    }

    public int sampleMethod(int y)
    {
        return x + y;
    }
}
```

The code above is able to be parsed but cannot be compiled by `javac`. Becasue there is no package like `java.utillll.*`.


- [ ] Figure out what errors [sensibility](https://github.com/naturalness/sensibility) repairs: syntax or compilation errors?
- [ ] What error do other tools focus on?
- [ ] Extract larger dataset of files with syntax errors.
- [ ] Mail to SNIC.

## Backlog

- [ ] Extract source files with compilation errors from BlueJ.
- [ ] Extract source files with compilation errors from Travis CI.

## Done
### Week 03
- [x] Partially reproduce the experiment of [LR parser-based error recovery experiment](https://github.com/softdevteam/error_recovery_experiment)
    - [x] Extract small subset (2,000) of source files with syntax errors
    - [x] Change the code to make tools generate "recovery suggestion messages" as shown in their paper.
    - [x] Able to generate some statistics of repair results
    - [x] Understand how this tool decides whether a repair is successful: just use parser.








### Week 01-02
- [x] Get access to BlueJ repository
- [x] Send initial research plans to supervisors and get feedback