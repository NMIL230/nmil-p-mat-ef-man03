
########################################################################################### 

BLACKLIST_SET = ["304run0","304run4","306run1"] # session IDs to be ignored in training
DEFAULT_HELDOUT_SET = ["406run9", "307run0","307run5","301run2","405run1","411run4","404run5","305run3","411run2","404run8"]


################################################################################################


METRICS_DICT = {  # ComplexSpan
    "CorsiComplex_correct_w_len_10": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_2": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_3": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_4": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_5": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_6": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_7": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_8": {"type": "binarySpan"},
    "CorsiComplex_correct_w_len_9": {"type": "binarySpan"},
    "CorsiComplex_reaction_time": {"type": "timing"},
    # Countermanding
    "Countermanding_is_correct": {"type": "binary"},
    "Countermanding_reaction_time": {"type": "timing"},
    # Cancellation
    "D2_hit_accuracy": {
        "type": "binary"
    },  # was initially beta but decided to model it as binomial
    # PasatPlus
    "PasatPlus_correctly_answered": {"type": "binary"},
    "PasatPlus_reaction_time": {"type": "timing"},
    # RunningSpan
    "RunningSpan_correct_w_len_1": {"type": "binary"},
    "RunningSpan_correct_w_len_2": {"type": "binary"},
    "RunningSpan_correct_w_len_3": {"type": "binary"},
    "RunningSpan_reaction_time": {"type": "timing"},
    # SimpleSpan
    "SimpleSpan_correct_w_len_10": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_2": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_3": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_4": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_5": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_6": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_7": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_8": {"type": "binarySpan"},
    "SimpleSpan_correct_w_len_9": {"type": "binarySpan"},
    "SimpleSpan_reaction_time": {"type": "timing"},
    # NumericalStroop
    "Stroop_correctly_answered": {"type": "binary"},
    "Stroop_reaction_time": {"type": "timing"},
    
    #MAGNITUDE COMPARISON
    "MagnitudeComparison_answered_correctly": {"type": "binary"},
    
    #NUMBERLINE
    "NumberLine_answered_within_correctness_threshold": {"type": "binary"},
    #RULESWITCH
    "RuleSwitch_reaction_time": {"type": "timing"},
    
    #FLANKER
    
    "Flanker_reaction_time": {"type": "timing"},
}


RELEVANT_METRICS = [  # ComplexSpan
    "CorsiComplex_correct_w_len_10",
    "CorsiComplex_correct_w_len_2",
    "CorsiComplex_correct_w_len_3",
    "CorsiComplex_correct_w_len_4",
    "CorsiComplex_correct_w_len_5",
    "CorsiComplex_correct_w_len_6",
    "CorsiComplex_correct_w_len_7",
    "CorsiComplex_correct_w_len_8",
    "CorsiComplex_correct_w_len_9",
    # Countermanding
    "Countermanding_reaction_time",
    # Cancellation
    "D2_hit_accuracy",
    # PasatPlus
    "PasatPlus_correctly_answered",
    # RunningSpan (only length 1 or 2)
    "RunningSpan_correct_w_len_2",
    "RunningSpan_correct_w_len_3",
    # SimpleSpan
    "SimpleSpan_correct_w_len_10",
    "SimpleSpan_correct_w_len_2",
    "SimpleSpan_correct_w_len_3",
    "SimpleSpan_correct_w_len_4",
    "SimpleSpan_correct_w_len_5",
    "SimpleSpan_correct_w_len_6",
    "SimpleSpan_correct_w_len_7",
    "SimpleSpan_correct_w_len_8",
    "SimpleSpan_correct_w_len_9",
    # NumericalStroop
    "Stroop_reaction_time",
]


SUMMARIZED_METRICS =["Countermanding_reaction_time","Stroop_reaction_time","SimpleSpan","CorsiComplex",
          "PasatPlus_correctly_answered","RunningSpan_correct_w_len_2","RunningSpan_correct_w_len_3","D2_hit_accuracy"]

SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = {
                            'SimpleSpan': "Simple Span psiTheta",
                            'CorsiComplex': "Complex Span psiTheta",
                          'Countermanding_reaction_time': "Countermanding Mean",
                            'D2_hit_accuracy': "Cancellation p",
                            "PasatPlus_correctly_answered":"PASATPlus p",
                            'RunningSpan_correct_w_len_2': "RS-2 p",
                            'RunningSpan_correct_w_len_3': "RS-3 p",
                            "Stroop_reaction_time":"Stroop mean"
                            }
ALL_METRICS_MOMENTS_LABEL_DICT = {
                "CorsiComplex_param1": 'Complex Span psiTheta',
                "CorsiComplex_param2": 'Complex Span psiSigma',
                "Countermanding_reaction_time_param1": 'Countermanding Mean',
                "Countermanding_reaction_time_param2": 'Countermanding StD',
                "D2_hit_accuracy_param1": 'Cancellation',
                "PasatPlus_correctly_answered_param1": 'PASAT',
                'RunningSpan_correct_w_len_1_param1': 'RS 1',
                'RunningSpan_correct_w_len_2_param1': 'RS 2',
                'SimpleSpan_param1': 'Simple Span psiTheta',
                'SimpleSpan_param2': 'Simple Span psiSigma',
                "Stroop_reaction_time_param1": 'Stroop Mean',
                "Stroop_reaction_time_param2": 'Stroop StD'
            }

ALL_METRICS = [
    "CorsiComplex_correct_w_len_10",
    "CorsiComplex_correct_w_len_2",
    "CorsiComplex_correct_w_len_3",
    "CorsiComplex_correct_w_len_4",
    "CorsiComplex_correct_w_len_5",
    "CorsiComplex_correct_w_len_6",
    "CorsiComplex_correct_w_len_7",
    "CorsiComplex_correct_w_len_8",
    "CorsiComplex_correct_w_len_9",
    "CorsiComplex_reaction_time",
    "Countermanding_is_correct",
    "Countermanding_reaction_time",
    "D2_hit_accuracy",
    "PasatPlus_correctly_answered",
    "PasatPlus_reaction_time",
    "RunningSpan_correct_w_len_2",
    "RunningSpan_correct_w_len_3",
    "RunningSpan_reaction_time",
    "SimpleSpan_correct_w_len_10",
    "SimpleSpan_correct_w_len_2",
    "SimpleSpan_correct_w_len_3",
    "SimpleSpan_correct_w_len_4",
    "SimpleSpan_correct_w_len_5",
    "SimpleSpan_correct_w_len_6",
    "SimpleSpan_correct_w_len_7",
    "SimpleSpan_correct_w_len_8",
    "SimpleSpan_correct_w_len_9",
    "SimpleSpan_reaction_time",
    "Stroop_correctly_answered",
    "Stroop_reaction_time",
]

# Ensure these variables have been declared:
'''
["METRICS_DICT", "RELEVANT_METRICS", "SUMMARIZED_METRICS",
                      "SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT", "ALL_METRICS_MOMENTS_LABEL_DICT",
                      "ALL_METRICS", "OUTLIER_HELDOUT_SESSIONS", "DEFAULT_HELDOUT_SET"]
'''
