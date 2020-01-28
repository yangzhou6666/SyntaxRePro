# Project Name
A comparison of self-healing parsers and machine learning for repairing syntax error

# Progress

## On Going
- [ ] Literature Review: define What is "Syntax Errors".
- [ ] Literature Review: What errors do these tools focus on?
- [ ] Write a short report about last two questions.
- [ ] Mail to SNIC to Handle User Agreement.

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

## Backlog

- [ ] Extract source files with compilation errors from BlueJ.
- [ ] Extract source files with compilation errors from Travis CI.

## Done
### Week 04


The code above is able to be parsed but cannot be compiled by `javac`. Becasue there is no package like `java.utillll.*`.



- [x] Extract larger dataset of files with syntax errors.
- [x] Have a Nice meeting with Earl and Martin.
- [x] Read Papers: [Targeted Example Generation for Compilation Errors](https://dblp.org/rec/html/conf/kbse/AhmedSSK19)

The tool in this paper does not repair syntax errors, but provides feedback for students. Unlike the other tools, the feedback is not error messages or generated repairing suggestions, but real fixes done by other students for similar errors. We can learn from the paper that:
* The idea of providing real fixes as feedback.
* Ways to measure how the tool helps student. It could be useful to your "*automatic feedback generation*" project.
* Generalize error messages to error types and error groups. We can also apply similar methodology in Java.

### Week 03
- [x] Partially reproduce the experiment of [LR parser-based error recovery experiment](https://github.com/softdevteam/error_recovery_experiment)
    - [x] Extract small subset (2,000) of source files with syntax errors
    - [x] Change the code to make tools generate "recovery suggestion messages" as shown in their paper.
    - [x] Able to generate some statistics of repair results
    - [x] Understand how this tool decides whether a repair is successful: just use parser.








### Week 01-02
- [x] Get access to BlueJ repository
- [x] Send initial research plans to supervisors and get feedback