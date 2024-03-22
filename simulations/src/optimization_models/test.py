# import numpy as np
# from warnings import warn


# options = {
#     "t_od": {
#         "use_in_vec": True,
#         "min_t_to_skm": 0.2,
#         "max_t_to_skm": 1.5,
#         "t_to_skm": 1, # must be bigger than min_t_to_skm
#         "t_cut_off": 0
#     },
#     "t_skm": {
#         "use_in_vec": False,
#         "max_var": 0.5,
#         "skm_freq": 4,
#         "skm_at_threshold": False,
#         "custom_skms": [60394.5, 60395, 60455]
#     }
# }

# threshold, duration = 3, 14
# start = 60390
# threshold = start + threshold
# duration = start + duration


# def get_design_vector_dict():

#     t_to_skm = options["t_od"]["t_to_skm"]
#     min_t_to_skm = options["t_od"]["min_t_to_skm"]
#     max_t_to_skm = options["t_od"]["max_t_to_skm"]
#     skm_freq = options["t_skm"]["skm_freq"]

#     skms = np.arange(start, duration, skm_freq)

#     # Remove values smaller than the threshold in the first list
#     design_vector_dict = dict()
#     for key, value in options.items():
#         if key == "t_od":
#             list = skms-t_to_skm
#         if key == "t_skm":
#             list = skms
#         design_vector_dict[key] = [x for x in list if x >= threshold]

#         if key == "t_skm":
#             if value["custom_skms"] is not None:

#                 skm_list = value["custom_skms"]
#                 if not all(skm_list[i] <= skm_list[i + 1] for i in range(len(skm_list) - 1)):
#                     warn(f'Custom SKMs in list are not chronological order, automatically sorted', RuntimeWarning)
#                     skm_list = sorted(skm_list)
#                 design_vector_dict[key] = skm_list
#                 design_vector_dict["t_od"] = [t_skm-t_to_skm for t_skm in skm_list]

#             if not value["skm_at_threshold"]:
#                 for i, epoch in enumerate(design_vector_dict[key]):
#                     if design_vector_dict[key][i] == threshold:
#                         design_vector_dict[key].remove(design_vector_dict[key][i])

#             for key, value in design_vector_dict.items():
#                 for epoch in value:
#                     if epoch < start:
#                         warn(f'Epoch {epoch} of {key} has value that is before minimum start epoch of MJD {start}', RuntimeWarning)
#                     if epoch > duration:
#                         warn(f'Epoch {epoch} of {key} has value that is after final duration epoch of MJD {duration}', RuntimeWarning)

#                 design_vector_dict[key] = [x for x in design_vector_dict[key] if x >= threshold and x<=duration]

#     # Some fault handling
#     if t_to_skm > skm_freq:
#         raise ValueError('Orbit determination of next SKM happens before current SKM')

#     if t_to_skm < min_t_to_skm:
#         raise ValueError('OD time to next SKM is smaller than required minimum')

#     if max_t_to_skm < t_to_skm:
#         raise ValueError('Maximum time to next SKM is smaller than currently set time to next SKM')


#     for i in range(len(design_vector_dict["t_skm"])):
#         if design_vector_dict["t_skm"][i] > design_vector_dict["t_od"][i+1]:
#             raise ValueError('Current t_od is smaller than previous SKM epoch')

#     if max_t_to_skm < t_to_skm:
#         raise ValueError('Maximum time to next SKM is smaller than currently set time to next SKM')

#     print("DESIGN VECTOR DICT: ", design_vector_dict)
#     return design_vector_dict


# def get_design_vector(design_vector_dict):

#     design_vector = []
#     for key, value in options.items():

#         if value["use_in_vec"]:
#             design_vector.extend(design_vector_dict[key])

#     print(design_vector)
#     return design_vector


# def get_bounds(design_vector_dict):

#     bounds_dict = dict()
#     for key, value in options.items():
#         skm_list = np.array(design_vector_dict["t_skm"])
#         t_od_list = np.array(design_vector_dict["t_od"])
#         if value["use_in_vec"]:
#             if key == "t_skm":

#                 bounds_dict[key] = list(zip(skm_list-options[key]["max_var"], skm_list+options[key]["max_var"]))

#             if key == "t_od":
#                 upper_bounds = np.array([abs(x - y) for x, y in zip(t_od_list, skm_list)])
#                 bounds_dict[key] = list(zip(skm_list-options[key]["max_t_to_skm"], skm_list-options[key]["min_t_to_skm"]))

#     if any(skm_list) == threshold:
#         bounds_dict["t_od"] = bounds_dict["t_od"][1:]

#     return bounds_dict






# def get_observation_windows(design_vector_dict):

#     observation_windows = [(start, threshold)]
#     a = 0
#     if len(design_vector_dict["t_skm"]) > len(design_vector_dict["t_od"]):
#         a = 1
#     observation_windows.extend([(design_vector_dict["t_od"][i], design_vector_dict["t_skm"][i+a]) for i in range(len(design_vector_dict["t_od"]))])

#     return observation_windows


# print(get_observation_windows(get_design_vector_dict()))

# print(get_design_vector(get_design_vector_dict()))

# print(get_bounds(get_design_vector_dict()))


import matplotlib.pyplot as plt
import numpy as np

data = {
    "point_mass": {
        "1": [
            6.099493670342755,
            418.9031059741974
        ],
        "1.5": [
            1.8747696316995026,
            364.94822931289673
        ],
        "2": [
            0.9230654815090421,
            304.23648405075073
        ],
        "2.5": [
            0.7487391188757612,
            275.8014669418335
        ],
        "3": [
            2.5160195255201163,
            284.1703951358795
        ],
        "3.5": [
            4.308090074065645,
            262.5213887691498
        ],
        "4": [
            9.780675917340947,
            281.28582978248596
        ],
        "4.5": [
            10.366853504258435,
            231.98801612854004
        ],
        "5": [
            24.627594089690696,
            239.09923887252808
        ]
    },
    "point_mass_srp": {
        "1": [
            0.43258911061480126,
            427.22055983543396
        ],
        "1.5": [
            0.1974304869160795,
            370.2760112285614
        ],
        "2": [
            0.08625908520428527,
            312.5080769062042
        ],
        "2.5": [
            0.09321623106205039,
            285.45557403564453
        ],
        "3": [
            0.23157419785810746,
            298.64599990844727
        ],
        "3.5": [
            0.4302184400069429,
            262.97373247146606
        ],
        "4": [
            0.8418177344467838,
            267.90875935554504
        ],
        "4.5": [
            0.8697649480305687,
            237.10099959373474
        ],
        "5": [
            2.6719102534392767,
            241.01633071899414
        ]
    },
    "spherical_harmonics": {
        "1": [
            6.022476723872688,
            1106.5791456699371
        ],
        "1.5": [
            1.9101921450110753,
            1018.6878905296326
        ],
        "2": [
            0.913000211536478,
            896.0826709270477
        ],
        "2.5": [
            0.7366424918087955,
            834.6297001838684
        ],
        "3": [
            2.710399982809485,
            861.54230260849
        ],
        "3.5": [
            4.222456796753445,
            779.9468185901642
        ],
        "4": [
            9.80206095699407,
            795.4987404346466
        ],
        "4.5": [
            10.087847412462978,
            723.212153673172
        ],
        "5": [
            24.676814615699662,
            742.1750733852386
        ]
    },
    "spherical_harmonics_srp": {
        "1": [
            0.47469231551794766,
            1120.3434882164001
        ],
        "1.5": [
            0.1743726644570397,
            1030.2374758720398
        ],
        "2": [
            0.09141442650810004,
            902.8569805622101
        ],
        "2.5": [
            0.07149375081518565,
            841.0781984329224
        ],
        "3": [
            0.2104953780064088,
            871.4050185680389
        ],
        "3.5": [
            0.31394425585427493,
            789.567862033844
        ],
        "4": [
            0.6933058510595228,
            803.0127635002136
        ],
        "4.5": [
            0.9485694558652384,
            729.622535943985
        ],
        "5": [
            2.2231198383235906,
            768.1847512722015
        ]
    }
}



import numpy as np

# # Given dictionary
# data = {
#     "point_mass": {
#         "1": [6.099493670342755, 418.9031059741974],
#         "1.5": [1.8747696316995026, 364.94822931289673],
#         "2": [0.9230654815090421, 304.23648405075073],
#         "2.5": [0.7487391188757612, 275.8014669418335],
#         "3": [2.5160195255201163, 284.1703951358795],
#         "3.5": [4.308090074065645, 262.5213887691498],
#         "4": [9.780675917340947, 281.28582978248596],
#         "4.5": [10.366853504258435, 231.98801612854004],
#         "5": [24.627594089690696, 239.09923887252808]
#     },
#     # Remaining keys...
# }

# Initialize dictionary to store mean and standard deviation for each value
stats = {}

# Iterate through the dictionary and calculate mean and standard deviation for each value
for key, sub_dict in data.items():
    stats[key] = {}
    for sub_key, value_list in sub_dict.items():
        mean_value = np.mean(value_list)
        std_dev_value = np.std(value_list)
        stats[key][sub_key] = {'mean': mean_value, 'std_dev': std_dev_value}

# Print the statistics dictionary
print(stats)
import os
import json
# data = {'point_mass': {0.2: [0.530908778438087, 155.03871512413025], 0.5: [0.38867258665653, 128.8769063949585], 0.8: [0.3538945426975028, 128.62496399879456], 1: [0.3915342384758657, 128.74781894683838], 1.2: [0.4140709980184972, 146.72685384750366]}, 'point_mass_srp': {0.2: [0.04183617217700521, 166.90307211875916], 0.5: [0.032455954078276436, 142.97971439361572], 0.8: [0.029061034544354807, 132.5707700252533], 1: [0.028960489163399213, 132.97550344467163], 1.2: [0.04001460917776998, 135.95449376106262]}, 'spherical_harmonics': {0.2: [0.5213456156903742, 443.2585029602051], 0.5: [0.3901441403368683, 398.5587854385376], 0.8: [0.35669061454693046, 410.7251400947571], 1: [0.3872732675548973, 415.48724579811096], 1.2: [0.4218262636491343, 423.679625749588]}, 'spherical_harmonics_srp': {0.2: [0.04330157607104699, 459.18675899505615], 0.5: [0.033292318178613604, 410.80441427230835], 0.8: [0.029833771274202787, 440.08425283432007], 1: [0.036616285639865465, 464.6441898345947], 1.2: [0.037211159779183656, 477.1388876438141]}}

# # Save the dictionary to a JSON file
# import json
# import os
# file_path =+".\simulations\src\optimization_models\data.json"
file_path = os.path.join( os.path.dirname(__file__), "TESTS.json")
with open(file_path, 'w') as json_file:
    json.dump(stats, json_file, indent=4)

# groups = list(data.keys())
# inner_keys = list(data[groups[0]].keys())
# num_groups = len(groups)

# fig, ax = plt.subplots(figsize=(8, 3))
# index = np.arange(len(inner_keys))
# bar_width = 0.2  # Adjust the width as needed

# # Center the bars around each xtick
# bar_offsets = np.arange(-(num_groups-1)/2, (num_groups-1)/2 + 1, 1) * bar_width

# for i in range(num_groups):
#     values = [data[groups[i]][inner_key][0] for inner_key in inner_keys]
#     ax.bar(index + bar_offsets[i], values, bar_width, label=str(groups[i]))

# ax.set_xlabel('Inner Keys')
# ax.set_ylabel('Values')
# ax.set_title('Bar Chart')
# ax.set_xticks(index)
# ax.set_xticklabels(inner_keys)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
# ax.set_yscale("log")
# plt.tight_layout()

# plt.show()