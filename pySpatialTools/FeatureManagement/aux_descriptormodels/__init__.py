
"""
Auxiliar descriptor models
--------------------------
This is a module which purpose is contain all the functions in parts that could
be useful for creating descriptor models.
"""

## Invocable characterizer functions
from characterizers import characterizer_1sh_counter, characterizer_summer,\
    characterizer_average

## Invocable reducer functions
from reducers import sum_reducer, avg_reducer

## Invocable add2result functions
from add2result_functions import sum_addresult_function,\
    append_addresult_function, replacelist_addresult_function

## Invocable completers
from completers import null_completer, weighted_completer,\
    sparse_dict_completer

## Invocable aggregation functions
from aggregation_functions import aggregator_1sh_counter, aggregator_summer,\
    aggregator_average

## Invocable featurenames functions
from featurenames_functions import counter_featurenames, array_featurenames

## Invocable out_formatter functions
from out_formatters import count_out_formatter, null_out_formatter
