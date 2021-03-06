
"""
Auxiliar descriptor models
--------------------------
This is a module which purpose is contain all the functions in parts that could
be useful for creating descriptor models.
"""

## Invocable characterizer functions
from characterizers import characterizer_1sh_counter,\
    characterizer_summer, characterizer_summer_array,\
    characterizer_summer_listdict, characterizer_summer_listarray,\
    characterizer_summer_arrayarray,\
    characterizer_average, characterizer_average_array,\
    characterizer_average_listdict, characterizer_average_listarray,\
    characterizer_average_arrayarray
from characterizers import characterizer_from_unitcharacterizer

## Invocable reducer functions
from reducers import sum_reducer, avg_reducer

## Invocable add2result functions
from add2result_functions import sum_addresult_function,\
    append_addresult_function, replacelist_addresult_function

## Invocable completers
from completers import null_completer, weighted_completer,\
    sparse_dict_completer, sparse_dict_completer_unknown,\
    null_completer_concatenator

## Invocable aggregation functions
from aggregation_functions import aggregator_1sh_counter, aggregator_summer,\
    aggregator_average

## Invocable featurenames functions
from featurenames_functions import counter_featurenames, array_featurenames,\
    list_featurenames, general_featurenames

## Invocable out_formatter functions
from out_formatters import count_out_formatter_general, null_out_formatter,\
    count_out_formatter_dict2array
