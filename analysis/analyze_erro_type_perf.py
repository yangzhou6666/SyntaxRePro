import csv
import matplotlib.pyplot as plt
import prettytable as pt
from tqdm import tqdm

deepfix = '''|            compiler.err.variable.not.allowed             |  1353  |  983   |    27.0%     |
|           compiler.err.array.dimension.missing           |  520   |  487   |     6.0%     |
|               compiler.err.bad.initializer               |   77   |   76   |     1.0%     |
|                  compiler.err.orphaned                   |  402   |  599   |    -49.0%    |
|       compiler.err.invalid.meth.decl.ret.type.req        |  5796  |  6355  |    -10.0%    |
|                 compiler.err.illegal.dot                 |   13   |   13   |     0.0%     |
|             compiler.err.enum.as.identifier              |   12   |   11   |     8.0%     |
|            compiler.err.illegal.start.of.expr            | 36739  | 33940  |     8.0%     |
|               compiler.err.else.without.if               |  3521  |  4245  |    -21.0%    |
|            compiler.err.illegal.start.of.stmt            |  1780  |  1509  |    15.0%     |
|              compiler.err.repeated.modifier              |   30   |   44   |    -47.0%    |
|            compiler.err.illegal.start.of.type            | 18733  | 19281  |    -3.0%     |
|             compiler.err.this.as.identifier              |   40   |   42   |    -5.0%     |
|             compiler.err.finally.without.try             |   15   |   17   |    -13.0%    |
|             compiler.err.dot.class.expected              |  5314  |  4902  |     8.0%     |
|                  compiler.err.expected                   | 216729 | 168254 |    22.0%     |
|                compiler.err.premature.eof                | 25806  | 25509  |     1.0%     |
|                  compiler.err.not.stmt                   | 33858  | 28787  |    15.0%     |
|                  compiler.err.expected3                  | 40843  | 51114  |    -25.0%    |
|                 compiler.err.local.enum                  |   3    |   4    |    -33.0%    |
|            compiler.err.varargs.and.receiver             |   1    |   1    |     0.0%     |
|            compiler.err.assert.as.identifier             |   8    |   12   |    -50.0%    |
|            compiler.err.mod.not.allowed.here             |   57   |   15   |    74.0%     |
|                  compiler.err.expected2                  |  3207  |  2236  |    30.0%     |
|              compiler.err.catch.without.try              |  201   |  231   |    -15.0%    |
|              compiler.err.class.not.allowed              |   1    |   1    |     0.0%     |
| compiler.err.try.without.catch.finally.or.resource.decls |  377   |  509   |    -35.0%    |
|                          Total                           | 395436 | 349177 |    12.0%     |
'''

rlassist = '''|            compiler.err.variable.not.allowed             |  3083  |  2782  |    9.76%     |
|           compiler.err.array.dimension.missing           |  961   |  962   |    -0.1%     |
|               compiler.err.bad.initializer               |  175   |  175   |     0.0%     |
|                  compiler.err.orphaned                   |  992   |  995   |    -0.3%     |
|       compiler.err.invalid.meth.decl.ret.type.req        |  9649  |  9207  |    4.58%     |
|                 compiler.err.illegal.dot                 |   14   |   18   |   -28.57%    |
|             compiler.err.enum.as.identifier              |   26   |   27   |    -3.85%    |
|            compiler.err.illegal.start.of.expr            | 84055  | 74006  |    11.96%    |
|               compiler.err.else.without.if               |  7916  |  7766  |    1.89%     |
|            compiler.err.illegal.start.of.stmt            |  3200  |  2613  |    18.34%    |
|              compiler.err.repeated.modifier              |   52   |   53   |    -1.92%    |
|            compiler.err.illegal.start.of.type            | 37477  | 33337  |    11.05%    |
|   compiler.err.cannot.create.array.with.type.arguments   |   2    |   2    |     0.0%     |
|             compiler.err.finally.without.try             |   40   |   37   |     7.5%     |
|             compiler.err.dot.class.expected              | 11065  | 11280  |    -1.94%    |
|                  compiler.err.expected                   | 425448 | 326060 |    23.36%    |
|                compiler.err.premature.eof                | 43779  | 17943  |    59.01%    |
|                  compiler.err.expected2                  |  5141  |  5058  |    1.61%     |
|                  compiler.err.expected3                  | 124020 | 114389 |    7.77%     |
|                 compiler.err.local.enum                  |   11   |   11   |     0.0%     |
|            compiler.err.varargs.and.receiver             |   1    |   1    |     0.0%     |
|             compiler.err.this.as.identifier              |  127   |  114   |    10.24%    |
|            compiler.err.assert.as.identifier             |   17   |   17   |     0.0%     |
|            compiler.err.mod.not.allowed.here             |   75   |   69   |     8.0%     |
|                  compiler.err.not.stmt                   | 73611  | 66643  |    9.47%     |
|              compiler.err.catch.without.try              |  475   |  451   |    5.05%     |
|              compiler.err.class.not.allowed              |   2    |   3    |    -50.0%    |
| compiler.err.try.without.catch.finally.or.resource.decls |  833   |  802   |    3.72%     |
|                          Total                           | 832247 | 674821 |    18.92%    |
'''

panic = '''|                      compiler.err.else.without.if                     |  8178  |  3651  |    55.36%    |
|                         compiler.err.expected3                        | 128047 | 64395  |    49.71%    |
|                         compiler.err.expected                         | 299317 | 324103 |    -8.28%    |
|                       compiler.err.premature.eof                      | 28422  | 37162  |   -30.75%    |
|              compiler.err.invalid.meth.decl.ret.type.req              |  7395  | 10236  |   -38.42%    |
|                         compiler.err.not.stmt                         | 64869  | 89373  |   -37.77%    |
|                   compiler.err.illegal.start.of.type                  | 31768  | 26140  |    17.72%    |
|                   compiler.err.illegal.start.of.expr                  | 43650  | 36427  |    16.55%    |
|                    compiler.err.dot.class.expected                    |  8603  |  8574  |    0.34%     |
|                         compiler.err.expected2                        |  6983  |  5924  |    15.17%    |
|        compiler.err.try.without.catch.finally.or.resource.decls       |  989   |  1144  |   -15.67%    |
|                   compiler.err.illegal.start.of.stmt                  |  2455  |  1749  |    28.76%    |
|                     compiler.err.unclosed.comment                     |  267   |   0    |    100.0%    |
|                  compiler.err.array.dimension.missing                 |  1020  |  1003  |    1.67%     |
|              compiler.err.preview.feature.disabled.plural             |  199   |   43   |    78.39%    |
|                     compiler.err.illegal.esc.char                     |  969   |   0    |    100.0%    |
|                   compiler.err.variable.not.allowed                   |  1736  |  1964  |   -13.13%    |
|                         compiler.err.orphaned                         |  907   |  219   |    75.85%    |
|                     compiler.err.catch.without.try                    |  555   |   43   |    92.25%    |
|                   compiler.err.int.number.too.large                   |  716   |  713   |    0.42%     |
|                    compiler.err.this.as.identifier                    |  106   |   67   |    36.79%    |
|                 compiler.err.underscore.as.identifier                 |  356   |  382   |    -7.3%     |
|                      compiler.err.bad.initializer                     |  185   |   7    |    96.22%    |
|                     compiler.err.repeated.modifier                    |   57   |  2048  |  -3492.98%   |
|             compiler.err.restricted.type.not.allowed.here             |  178   |  117   |    34.27%    |
|                        compiler.err.illegal.dot                       |  215   |  157   |    26.98%    |
|                     compiler.err.unclosed.str.lit                     |  132   |   0    |    100.0%    |
| compiler.err.illegal.array.creation.both.dimension.and.initialization |   54   |   37   |    31.48%    |
|                    compiler.err.finally.without.try                   |   52   |   6    |    88.46%    |
|                      compiler.err.wrong.receiver                      |   74   |   51   |    31.08%    |
|                   compiler.err.mod.not.allowed.here                   |   88   |   94   |    -6.82%    |
|             compiler.err.try.with.resources.expr.needs.var            |   72   |   56   |    22.22%    |
|                     compiler.err.malformed.fp.lit                     |   53   |   26   |    50.94%    |
|                  compiler.err.illegal.text.block.open                 |   71   |   0    |    100.0%    |
|                  compiler.err.initializer.not.allowed                 |   41   |   14   |    65.85%    |
|                   compiler.err.assert.as.identifier                   |   15   |   7    |    53.33%    |
|                       compiler.err.illegal.char                       |   16   |   9    |    43.75%    |
|           compiler.err.restricted.type.not.allowed.compound           |   47   |   48   |    -2.13%    |
|                        compiler.err.local.enum                        |   9    |   8    |    11.11%    |
|           compiler.err.invalid.lambda.parameter.declaration           |   10   |   88   |   -780.0%    |
|                    compiler.err.invalid.hex.number                    |   2    |   2    |     0.0%     |
|                    compiler.err.illegal.unicode.esc                   |   9    |   0    |    100.0%    |
|                    compiler.err.illegal.underscore                    |   20   |   16   |    20.0%     |
|                compiler.err.restricted.type.not.allowed               |   6    |   6    |     0.0%     |
|               compiler.err.illegal.line.end.in.char.lit               |   2    |   0    |    100.0%    |
|                     compiler.err.unclosed.char.lit                    |   4    |   0    |    100.0%    |
|                    compiler.err.enum.as.identifier                    |   33   |   24   |    27.27%    |
|                   compiler.err.varargs.and.receiver                   |   1    |   0    |    100.0%    |
|                     compiler.err.class.not.allowed                    |   2    |   3    |    -50.0%    |
|          compiler.err.cannot.create.array.with.type.arguments         |   2    |   0    |    100.0%    |
|                   compiler.err.varargs.must.be.last                   |   1    |   1    |     0.0%     |
|                   compiler.err.invalid.binary.number                  |   4    |   4    |     0.0%     |
|                                 Total                                 | 638962 | 616141 |    3.57%     |
'''

cpct_plus = '''|                         compiler.err.expected3                        | 96488  | 41782 |    56.7%     |
|                         compiler.err.expected                         | 224462 | 15228 |    93.22%    |
|                      compiler.err.else.without.if                     |  6167  |   21  |    99.66%    |
|                       compiler.err.premature.eof                      | 21422  |   2   |    99.99%    |
|              compiler.err.invalid.meth.decl.ret.type.req              |  5447  |  5712 |    -4.87%    |
|                         compiler.err.not.stmt                         | 48718  |  567  |    98.84%    |
|                   compiler.err.illegal.start.of.type                  | 23664  |  7326 |    69.04%    |
|                   compiler.err.illegal.start.of.expr                  | 32702  |  6413 |    80.39%    |
|                         compiler.err.expected2                        |  5262  |   8   |    99.85%    |
|                    compiler.err.dot.class.expected                    |  6499  |   2   |    99.97%    |
|        compiler.err.try.without.catch.finally.or.resource.decls       |  753   |   18  |    97.61%    |
|                   compiler.err.illegal.start.of.stmt                  |  1887  |   0   |    100.0%    |
|                     compiler.err.unclosed.comment                     |  182   |   0   |    100.0%    |
|                  compiler.err.array.dimension.missing                 |  773   |   0   |    100.0%    |
|              compiler.err.preview.feature.disabled.plural             |  149   |   19  |    87.25%    |
|                     compiler.err.illegal.esc.char                     |  552   |   0   |    100.0%    |
|                   compiler.err.variable.not.allowed                   |  1347  |   0   |    100.0%    |
|                         compiler.err.orphaned                         |  645   |   2   |    99.69%    |
|                     compiler.err.catch.without.try                    |  430   |   1   |    99.77%    |
|                   compiler.err.int.number.too.large                   |  610   |  293  |    51.97%    |
|                    compiler.err.this.as.identifier                    |   83   |   0   |    100.0%    |
|                 compiler.err.underscore.as.identifier                 |  242   |  288  |   -19.01%    |
|                      compiler.err.bad.initializer                     |  139   |   0   |    100.0%    |
|                     compiler.err.repeated.modifier                    |   37   |   47  |   -27.03%    |
|             compiler.err.restricted.type.not.allowed.here             |  155   |  124  |    20.0%     |
|                        compiler.err.illegal.dot                       |  141   |   0   |    100.0%    |
|                     compiler.err.unclosed.str.lit                     |  105   |   0   |    100.0%    |
| compiler.err.illegal.array.creation.both.dimension.and.initialization |   39   |   0   |    100.0%    |
|                    compiler.err.finally.without.try                   |   40   |   0   |    100.0%    |
|                      compiler.err.wrong.receiver                      |   62   |   3   |    95.16%    |
|                   compiler.err.mod.not.allowed.here                   |   63   |   6   |    90.48%    |
|             compiler.err.try.with.resources.expr.needs.var            |   50   |   0   |    100.0%    |
|                     compiler.err.malformed.fp.lit                     |   31   |   21  |    32.26%    |
|                  compiler.err.illegal.text.block.open                 |   66   |   0   |    100.0%    |
|                  compiler.err.initializer.not.allowed                 |   38   |   1   |    97.37%    |
|                   compiler.err.assert.as.identifier                   |   9    |   8   |    11.11%    |
|                       compiler.err.illegal.char                       |   14   |   7   |    50.0%     |
|           compiler.err.restricted.type.not.allowed.compound           |   47   |   31  |    34.04%    |
|                        compiler.err.local.enum                        |   8    |   7   |    12.5%     |
|           compiler.err.invalid.lambda.parameter.declaration           |   9    |   0   |    100.0%    |
|                    compiler.err.invalid.hex.number                    |   2    |   2   |     0.0%     |
|                    compiler.err.illegal.unicode.esc                   |   9    |   0   |    100.0%    |
|                    compiler.err.illegal.underscore                    |   19   |   19  |     0.0%     |
|                compiler.err.restricted.type.not.allowed               |   5    |   5   |     0.0%     |
|               compiler.err.illegal.line.end.in.char.lit               |   2    |   0   |    100.0%    |
|                     compiler.err.unclosed.char.lit                    |   4    |   0   |    100.0%    |
|                    compiler.err.enum.as.identifier                    |   23   |   0   |    100.0%    |
|                   compiler.err.varargs.and.receiver                   |   1    |   0   |    100.0%    |
|                     compiler.err.class.not.allowed                    |   2    |   0   |    100.0%    |
|          compiler.err.cannot.create.array.with.type.arguments         |   2    |   0   |    100.0%    |
|                   compiler.err.varargs.must.be.last                   |   1    |   1   |     0.0%     |
|             compiler.err.cannot.create.array.with.diamond             |   0    |   7   |     NAN      |
|                                 Total                                 | 479607 | 77971 |    83.74%    |
'''
def analzye_perf_on_err_tpyes(data):

    error_types = ['expected', 'expected3', 'illegal.start.of.expr', 'not.stmt', 'illegal.start.of.type', 'premature.eof', 'dot.class.expected', 'invalid.meth.decl.ret.type.req', 'else.without.if', 'expected2']

    tb = pt.PrettyTable()
    tb.field_names = ['Error Tpye', 'Repair Rate']

    # replace ' ' with ''
    data = data.replace(' ', '')
    data = data.split('|\n')
    repair_rate_by_token = {}

    for need_error_type in tqdm(error_types):
        for item in data:
            data_by_token = item.split('|')[1:]
            try:
                err_type = data_by_token[0][13:]
                if err_type == need_error_type:
                    tb.add_row([err_type, data_by_token[3]])
                    # print(data_by_token)
            except:
                continue

    print(tb)
    
analzye_perf_on_err_tpyes(cpct_plus)
analzye_perf_on_err_tpyes(panic)
analzye_perf_on_err_tpyes(rlassist)
analzye_perf_on_err_tpyes(deepfix)