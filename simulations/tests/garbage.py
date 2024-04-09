# # # from datetime import datetime

# # # current_time = datetime.now()

# # # # Format the current time as a string
# # # current_time_string = datetime.now().strftime("%d%m_%H%M")

# # # print(current_time_string)  # Output: "2024-03-29 14:30:00"

# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # data = {"0": {
# # #    "threshold": 60393,
# # #    "skm_to_od_duration": 3,
# # #    "duration": 28,
# # #    "factor": 1,
# # #    "maxiter": 60,
# # #    "initial_design_vector": [
# # #       1.0,
# # #       1.0,
# # #       1.0,
# # #       1.0,
# # #       1.0,
# # #       1.0
# # #    ],
# # #    "model": {
# # #       "dynamic": {
# # #          "model_type": "HF",
# # #          "model_name": "PM",
# # #          "model_number": 0
# # #       },
# # #       "truth": {
# # #          "model_type": "HF",
# # #          "model_name": "PM",
# # #          "model_number": 0
# # #       }
# # #    },
# # #    "history": {
# # #       "design_vector": {
# # #          "0": [
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0
# # #          ],
# # #          "1": [
# # #             1.0,
# # #             1.05,
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0
# # #          ],
# # #          "2": [
# # #             1.0,
# # #             1.05,
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0
# # #          ],
# # #          "3": [
# # #             1.0,
# # #             1.05,
# # #             1.0,
# # #             1.0,
# # #             1.0,
# # #             1.0
# # #          ],
# # #          "4": [
# # #             1.0264060356652946,
# # #             1.036488340192044,
# # #             0.9503772290809329,
# # #             0.972633744855967,
# # #             1.036488340192044,
# # #             0.9911179698216734
# # #          ],
# # #          "5": [
# # #             1.0366750495351311,
# # #             1.0506782502667276,
# # #             0.9977461515012951,
# # #             0.9619913122999539,
# # #             1.0506782502667271,
# # #             0.9876638469745462
# # #          ],
# # #          "6": [
# # #             1.043660773256109,
# # #             1.060331250317533,
# # #             0.9973168470253517,
# # #             0.9547515622618501,
# # #             0.993664583650866,
# # #             0.9853141035411268
# # #          ],
# # #          "7": [
# # #             1.0125353519957996,
# # #             1.0755033954851052,
# # #             0.9914841910955305,
# # #             0.9433724533861707,
# # #             1.0366145065962165,
# # #             0.9754805331165639
# # #          ],
# # #          "8": [
# # #             1.0125353519957996,
# # #             1.0755033954851052,
# # #             0.9914841910955305,
# # #             0.9433724533861707,
# # #             1.0366145065962165,
# # #             0.9754805331165639
# # #          ],
# # #          "9": [
# # #             1.0125353519957996,
# # #             1.0755033954851052,
# # #             0.9914841910955305,
# # #             0.9433724533861707,
# # #             1.0366145065962165,
# # #             0.9754805331165639
# # #          ],
# # #          "10": [
# # #             1.0125353519957996,
# # #             1.0755033954851052,
# # #             0.9914841910955305,
# # #             0.9433724533861707,
# # #             1.0366145065962165,
# # #             0.9754805331165639
# # #          ],
# # #          "11": [
# # #             1.0125353519957996,
# # #             1.0755033954851052,
# # #             0.9914841910955305,
# # #             0.9433724533861707,
# # #             1.0366145065962165,
# # #             0.9754805331165639
# # #          ],
# # #          "12": [
# # #             1.0375942069431952,
# # #             1.1302848096141855,
# # #             0.9470117252078274,
# # #             0.9375744586329815,
# # #             0.9995882694526252,
# # #             0.95627266229523
# # #          ],
# # #          "13": [
# # #             1.0259206884157575,
# # #             1.144618304828476,
# # #             0.9280210282467143,
# # #             0.9405474739392266,
# # #             1.0779101471966797,
# # #             0.9481117597219408
# # #          ],
# # #          "14": [
# # #             1.0259206884157575,
# # #             1.144618304828476,
# # #             0.9280210282467143,
# # #             0.9405474739392266,
# # #             1.0779101471966797,
# # #             0.9481117597219408
# # #          ],
# # #          "15": [
# # #             1.0259206884157575,
# # #             1.144618304828476,
# # #             0.9280210282467143,
# # #             0.9405474739392266,
# # #             1.0779101471966797,
# # #             0.9481117597219408
# # #          ],
# # #          "16": [
# # #             1.0361611032534894,
# # #             1.1351281980479642,
# # #             0.9452168287760125,
# # #             0.8883591527110115,
# # #             1.0704332094331535,
# # #             0.9205547664829321
# # #          ],
# # #          "17": [
# # #             1.0361611032534894,
# # #             1.1351281980479642,
# # #             0.9452168287760125,
# # #             0.8883591527110115,
# # #             1.0704332094331535,
# # #             0.9205547664829321
# # #          ],
# # #          "18": [
# # #             1.0483542955251033,
# # #             1.213322381856245,
# # #             0.8610146015589191,
# # #             0.8828781130363894,
# # #             1.06073602129079,
# # #             0.9377225799188844
# # #          ],
# # #          "19": [
# # #             1.0191269678215402,
# # #             1.1725728855560504,
# # #             0.8868678968790591,
# # #             0.8693379716419665,
# # #             1.133884018752642,
# # #             0.9561191534231948
# # #          ],
# # #          "20": [
# # #             1.0191269678215402,
# # #             1.1725728855560504,
# # #             0.8868678968790591,
# # #             0.8693379716419665,
# # #             1.133884018752642,
# # #             0.9561191534231948
# # #          ],
# # #          "21": [
# # #             1.0061869596862527,
# # #             1.226958625841562,
# # #             0.8676509085567272,
# # #             0.8967041698411178,
# # #             1.121897140799983,
# # #             0.8751207701294725
# # #          ],
# # #          "22": [
# # #             1.0205138620224141,
# # #             1.2415924531567657,
# # #             0.8749120837596585,
# # #             0.8162529531512424,
# # #             1.0952677992859623,
# # #             0.9022209391107037
# # #          ],
# # #          "23": [
# # #             1.063285799725515,
# # #             1.2558994198133082,
# # #             0.836950214541686,
# # #             0.8080633240674115,
# # #             1.173856553218878,
# # #             0.8652314978746236
# # #          ],
# # #          "24": [
# # #             1.063285799725515,
# # #             1.2558994198133082,
# # #             0.836950214541686,
# # #             0.8080633240674115,
# # #             1.173856553218878,
# # #             0.8652314978746236
# # #          ],
# # #          "25": [
# # #             1.063285799725515,
# # #             1.2558994198133082,
# # #             0.836950214541686,
# # #             0.8080633240674115,
# # #             1.173856553218878,
# # #             0.8652314978746236
# # #          ],
# # #          "26": [
# # #             1.0133269721003904,
# # #             1.286619576435404,
# # #             0.7588490337573963,
# # #             0.7965950767460237,
# # #             1.2215624382124943,
# # #             0.8576901388903297
# # #          ],
# # #          "27": [
# # #             1.0133269721003904,
# # #             1.286619576435404,
# # #             0.7588490337573963,
# # #             0.7965950767460237,
# # #             1.2215624382124943,
# # #             0.8576901388903297
# # #          ],
# # #          "28": [
# # #             1.0534275291059467,
# # #             1.3582101763054988,
# # #             0.7644405066191522,
# # #             0.687172445661238,
# # #             1.2370075286190287,
# # #             0.8525441695677483
# # #          ],
# # #          "29": [
# # #             1.0534275291059467,
# # #             1.3582101763054988,
# # #             0.7644405066191522,
# # #             0.687172445661238,
# # #             1.2370075286190287,
# # #             0.8525441695677483
# # #          ],
# # #          "30": [
# # #             1.0448752041981177,
# # #             1.3218499152936882,
# # #             0.7723858246503772,
# # #             0.7038467160939619,
# # #             1.3031999925207958,
# # #             0.7677843666121356
# # #          ],
# # #          "31": [
# # #             1.0740470936794604,
# # #             1.3883944021514703,
# # #             0.6767920977052695,
# # #             0.6986957619505958,
# # #             1.27696600935364,
# # #             0.7829608040213523
# # #          ],
# # #          "32": [
# # #             1.0219978291109892,
# # #             1.4343994801356683,
# # #             0.6612017526075715,
# # #             0.6663932703904127,
# # #             1.3328373106239142,
# # #             0.7590845817505152
# # #          ],
# # #          "33": [
# # #             1.0219978291109892,
# # #             1.4343994801356683,
# # #             0.6612017526075715,
# # #             0.6663932703904127,
# # #             1.3328373106239142,
# # #             0.7590845817505152
# # #          ],
# # #          "34": [
# # #             1.0219978291109892,
# # #             1.4343994801356683,
# # #             0.6612017526075715,
# # #             0.6663932703904127,
# # #             1.3328373106239142,
# # #             0.7590845817505152
# # #          ],
# # #          "35": [
# # #             1.0219978291109892,
# # #             1.4343994801356683,
# # #             0.6612017526075715,
# # #             0.6663932703904127,
# # #             1.3328373106239142,
# # #             0.7590845817505152
# # #          ],
# # #          "36": [
# # #             1.0219978291109892,
# # #             1.4343994801356683,
# # #             0.6612017526075715,
# # #             0.6663932703904127,
# # #             1.3328373106239142,
# # #             0.7590845817505152
# # #          ],
# # #          "37": [
# # #             1.0840652184485102,
# # #             1.5,
# # #             0.5067978023141677,
# # #             0.5389376589851818,
# # #             1.4244789135246845,
# # #             0.7397629878043501
# # #          ],
# # #          "38": [
# # #             1.0840652184485102,
# # #             1.5,
# # #             0.5067978023141677,
# # #             0.5389376589851818,
# # #             1.4244789135246845,
# # #             0.7397629878043501
# # #          ],
# # #          "39": [
# # #             1.0840652184485102,
# # #             1.5,
# # #             0.5067978023141677,
# # #             0.5389376589851818,
# # #             1.4244789135246845,
# # #             0.7397629878043501
# # #          ],
# # #          "40": [
# # #             1.0505650707619365,
# # #             1.5,
# # #             0.5283019614036062,
# # #             0.5151770512634359,
# # #             1.4382483678561655,
# # #             0.6589102150787258
# # #          ],
# # #          "41": [
# # #             1.0505650707619365,
# # #             1.5,
# # #             0.5283019614036062,
# # #             0.5151770512634359,
# # #             1.4382483678561655,
# # #             0.6589102150787258
# # #          ],
# # #          "42": [
# # #             1.0172186693628593,
# # #             1.5,
# # #             0.5,
# # #             0.533582717909757,
# # #             1.5,
# # #             0.6996751997242181
# # #          ],
# # #          "43": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "44": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "45": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "46": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "47": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "48": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "49": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "50": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "51": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "52": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "53": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "54": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "55": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "56": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "57": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "58": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ],
# # #          "59": [
# # #             1.0696548772860786,
# # #             1.5,
# # #             0.5,
# # #             0.5,
# # #             1.5,
# # #             0.6872476257360807
# # #          ]
# # #       },
# # #       "objective_value": {
# # #          "0": 2.2463962616204514,
# # #          "1": 2.087376371321291,
# # #          "2": 2.0788247091575114,
# # #          "3": 2.112384285001497,
# # #          "4": 2.030556099309441,
# # #          "5": 1.9882453398525697,
# # #          "6": 1.9926368122806557,
# # #          "7": 1.9113406651607177,
# # #          "8": 1.8690626826718644,
# # #          "9": 1.8821283505368829,
# # #          "10": 1.8824048191889102,
# # #          "11": 1.876043169312066,
# # #          "12": 1.8256205833461294,
# # #          "13": 1.8083523856693013,
# # #          "14": 1.7789062096569401,
# # #          "15": 1.8080865426610764,
# # #          "16": 1.7308005291646573,
# # #          "17": 1.7413464856693281,
# # #          "18": 1.6764590308310647,
# # #          "19": 1.666387685614776,
# # #          "20": 1.6826001393443568,
# # #          "21": 1.644827745943528,
# # #          "22": 1.5777321116549634,
# # #          "23": 1.5373637186037494,
# # #          "24": 1.5423983056063961,
# # #          "25": 1.5582332220031359,
# # #          "26": 1.4755595653787996,
# # #          "27": 1.4875860479722096,
# # #          "28": 1.3932959799772324,
# # #          "29": 1.3930592132177437,
# # #          "30": 1.3744397981694862,
# # #          "31": 1.3966307473915986,
# # #          "32": 1.3414602468465557,
# # #          "33": 1.3459434567048076,
# # #          "34": 1.3416425263612948,
# # #          "35": 1.3511199935266855,
# # #          "36": 1.3520745707036053,
# # #          "37": 1.2840018863803586,
# # #          "38": 1.2484664482433272,
# # #          "39": 1.256719860876987,
# # #          "40": 1.2545224243826771,
# # #          "41": 1.2684238714343379,
# # #          "42": 1.2338323213089428,
# # #          "43": 1.2512560908981583,
# # #          "44": 1.261599629747322,
# # #          "45": 1.2404106323455186,
# # #          "46": 1.257593008996446,
# # #          "47": 1.2420976773923513,
# # #          "48": 1.2445197741969287,
# # #          "49": 1.2343623907538337,
# # #          "50": 1.2563751194369472,
# # #          "51": 1.2533295831355173,
# # #          "52": 1.2355247665516167,
# # #          "53": 1.252814639833546,
# # #          "54": 1.2429632606768946,
# # #          "55": 1.2468555843716753,
# # #          "56": 1.2661449357700019,
# # #          "57": 1.2548953530966265,
# # #          "58": 1.2495921570204127,
# # #          "59": 1.258026412406305
# # #       }
# # #    },
# # #    "final_result": {
# # #       "x_optim": [
# # #          1.0696548772860786,
# # #          1.5,
# # #          0.5,
# # #          0.5,
# # #          1.5,
# # #          0.6872476257360807
# # #       ],
# # #       "observation_windows": [
# # #          [
# # #             60390,
# # #             60393
# # #          ],
# # #          [
# # #             60396.0,
# # #             60397.06965487728
# # #          ],
# # #          [
# # #             60400.06965487728,
# # #             60401.56965487728
# # #          ],
# # #          [
# # #             60404.56965487728,
# # #             60405.06965487728
# # #          ],
# # #          [
# # #             60408.06965487728,
# # #             60408.56965487728
# # #          ],
# # #          [
# # #             60411.56965487728,
# # #             60413.06965487728
# # #          ],
# # #          [
# # #             60416.06965487728,
# # #             60416.75690250302
# # #          ]
# # #       ],
# # #       "skm_epochs": [
# # #          60393,
# # #          60397.06965487728,
# # #          60401.56965487728,
# # #          60405.06965487728,
# # #          60408.56965487728,
# # #          60413.06965487728,
# # #          60416.75690250302
# # #       ],
# # #       "approx_annual_deltav": 16.399272876010762,
# # #       "reduction_percentage": -43.998018786818136,
# # #       "run_time": 29590.47538471222
# # #    }
# # # }}



# # # print([list(item["history"]["objective_value"].values()) for item in data.values()])

# # # # Extracting objective values into a 2D array
# # # objective_values = np.array([list(item['history']['objective_value'].values()) for item in data.values()]).T
# # # # Extracting design vectors into a 3D array
# # # design_vectors = np.array([list(item['history']['design_vector'].values()) for item in data.values()])

# # # print("Objective Values:")
# # # print(objective_values)

# # # print("\nDesign Vectors:")
# # # print(design_vectors)


# # # # # Extract objective function values
# # # # iteration_numbers = []
# # # # objective_functions = []
# # # # states = []

# # # # for iteration, values in data["history"].items():
# # # #     iteration_numbers.append(int(iteration))
# # # #     objective_functions.append(values["objective_function"])
# # # #     states.append(values["design_vector"])

# # # # Plot
# # # plt.plot(objective_values, marker='*', linestyle='-')
# # # plt.xlabel('Iteration')
# # # plt.ylabel(r'||$\Delta V$||')
# # # plt.title('Objective Function Value vs. Iteration')
# # # plt.grid(True)
# # # # plt.show()


# # # plt.plot(design_vectors[0], marker='*', linestyle='-', label=["T1", "T2", "T3", "T4", "T5", "T6"])
# # # plt.xlabel('Iteration')
# # # plt.ylabel('States')
# # # plt.title('State Evolution vs. Iteration')
# # # plt.grid(True)
# # # plt.show()

# # # import matplotlib.pyplot as plt

# # # # Example data (mean and standard deviation)
# # # mean_values = [1, 2, 3, 4, 5]
# # # std_values = [0.1, 0.2, 0.1, 0.3, 0.2]

# # # # Create x-values (assuming you have 5 data points)
# # # x_values = range(len(mean_values))

# # # # Create the plot
# # # plt.errorbar(x_values, mean_values, yerr=std_values, fmt='o', capsize=5)

# # # # Add labels and title
# # # plt.xlabel('Data Point')
# # # plt.ylabel('Mean Value')
# # # plt.title('Mean Values with Error Bars')

# # # # Show the plot
# # # plt.show()

# # # import numpy as np

# # # # Define inertial states (position and velocity vectors) of the satellite in the Earth-centered inertial frame
# # # satellite_position_inertial = np.array([x_satellite, y_satellite, z_satellite])
# # # satellite_velocity_inertial = np.array([vx_satellite, vy_satellite, vz_satellite])

# # # # Define the position vector from the Earth to the Moon
# # # earth_to_moon_position = moon_position_inertial - earth_position_inertial

# # # # Calculate the velocity vector of the Moon relative to the Earth
# # # moon_velocity_relative_to_earth = moon_velocity_inertial - earth_velocity_inertial

# # # # Calculate the angular velocity vector of the Earth-Moon system
# # # angular_velocity_earth_moon = np.cross(earth_to_moon_position, moon_velocity_relative_to_earth) / np.linalg.norm(earth_to_moon_position)**2

# # # # Construct the rotation matrix for transforming states from the inertial frame to the rotating frame
# # # rotation_matrix = np.array([[1, 0, 0],
# # #                             [0, np.cos(theta), -np.sin(theta)],
# # #                             [0, np.sin(theta), np.cos(theta)]])

# # # # Transform the states from the inertial frame to the rotating frame
# # # satellite_position_rotating = np.dot(rotation_matrix, satellite_position_inertial)
# # # satellite_velocity_rotating = np.dot(rotation_matrix, satellite_velocity_inertial - np.cross(angular_velocity_earth_moon, satellite_position_inertial))

# # # # satellite_position_rotating and satellite_velocity_rotating now contain the satellite states in the rotating frame

# import numpy as np

# # Example values for position vector of the Moon relative to the Earth
# x_moon = -279077.269273
# y_moon = 252757.216883
# z_moon = 145049.670755

# # Example values for velocity vector of the Moon relative to the Earth
# vx_moon = -0.719984
# vy_moon = -0.583631
# vz_moon = -0.297165

# # Define the position vector of the Moon relative to the Earth
# moon_position = np.array([x_moon, y_moon, z_moon])
# moon_velocity = np.array([vx_moon, vy_moon, vz_moon])
# rotation_axis = np.cross(moon_position, moon_velocity)
# rotation_rate_magnitude = np.linalg.norm(rotation_axis) / np.linalg.norm(moon_position)**2
# rotation_axis_unit = rotation_axis / np.linalg.norm(rotation_axis)
# second_axis = np.cross(moon_position, rotation_axis)

# first_axis = moon_position/np.linalg.norm(moon_position)
# second_axis = second_axis/np.linalg.norm(second_axis)
# third_axis = rotation_axis/np.linalg.norm(rotation_axis)

# # print("Rotation Axis Vector:", rotation_axis_unit)
# # print("Magnitude of Rotation Rate:", rotation_rate_magnitude)

# # print("First axis:", moon_position/np.linalg.norm(moon_position))
# # print("Second axis:", second_axis/np.linalg.norm(second_axis))
# # print("Third axis:", rotation_axis/np.linalg.norm(rotation_axis))


# # Define the rotation matrix (DCM) using the rotating frame axes
# rotating_frame_axes = np.array([
#     first_axis, second_axis, third_axis    # Third axis
# ])

# rotating_state = np.dot(rotating_frame_axes, moon_position)

# # Print the result
# print("State in rotating frame:", rotating_state)
# # print(second_axis/np.linalg.norm(second_axis), moon_velocity/np.linalg.norm(moon_velocity))


# # # # Example unit vector representing angular velocity
# # # angular_velocity_unit_vector = rotation_axis_unit
# # # azimuth_angle = np.arctan2(angular_velocity_unit_vector[1], angular_velocity_unit_vector[0])
# # # polar_angle = np.arcsin(angular_velocity_unit_vector[2])

# # # # Convert angles from radians to degrees for easier interpretation
# # # azimuth_angle_deg = np.degrees(azimuth_angle)
# # # polar_angle_deg = np.degrees(polar_angle)

# # # print("Azimuth Angle (Yaw) in degrees:", azimuth_angle_deg)
# # # print("Polar Angle (Pitch) in degrees:", polar_angle_deg)



# # # # Construct the transformation matrix (DCM) using the azimuth and polar angles
# # # cos_yaw = np.cos(azimuth_angle)
# # # sin_yaw = np.sin(azimuth_angle)
# # # cos_pitch = np.cos(polar_angle)
# # # sin_pitch = np.sin(polar_angle)

# # # # Define the elements of the transformation matrix (DCM)
# # # # Assuming z-y'-x'' sequence of rotations
# # # dcm = np.array([
# # #     [cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch],
# # #     [sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch],
# # #     [-sin_pitch, 0, cos_pitch]
# # # ])

# # # print("Transformation Matrix (DCM):")
# # # print(dcm)


# # # x_inertial = -279077.269273
# # # y_inertial = 252757.216883
# # # z_inertial = 145049.670755

# # # # Define the inertial state vector (example values)
# # # inertial_state = np.array([x_inertial, y_inertial, z_inertial])

# # # # Calculate the state vector in the rotating Earth-Moon frame
# # # rotating_frame_state = np.dot(dcm, inertial_state)
# # # print(np.linalg.norm(rotating_frame_state))

# # # print("State Vector in Rotating Frame:")
# # # print(rotating_frame_state)







# # # # Define the unit vector of rotation and the plane of rotation
# # # rotation_axis = np.array([0, 0, 1])  # Example rotation axis
# # # rotation_plane = inertial_state  # Example rotation plane

# # # # Calculate the cross product to obtain a third orthogonal vector
# # # orthogonal_vector = np.cross(rotation_axis, rotation_plane)
# # # # orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # Normalize

# # # # # Normalize the rotation axis
# # # # rotation_axis /= np.linalg.norm(rotation_axis)

# # # # Create the transformation matrix
# # # transformation_matrix = np.array([rotation_axis, orthogonal_vector, rotation_plane])

# # # # Define the vector you want to transform (example values)
# # # vector_to_transform = np.array([1.0, 0.0, 0.0])

# # # # Transform the vector to the desired frame
# # # transformed_vector = np.dot(transformation_matrix.T, vector_to_transform)

# # # print(transformation_matrix, transformed_vector)



# # import numpy as np

# # # Example rotation axis vector of moon rotation around earth (replace with your vector)
# # rotation_axis = np.array([0.02431277, -0.47725793, 0.87842686])

# # # Example position vector of moon w.r.t earth (replace with your vector)
# # moon_position = np.array([-279077.269273, 252757.216883, 145049.670755])

# # # Normalize the rotation axis vector
# # rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)

# # # Take the cross product of the rotation axis vector and the position vector of the Moon
# # cross_product = np.cross(rotation_axis_normalized, moon_position)

# # # Normalize the result
# # cross_product_normalized = cross_product / np.linalg.norm(cross_product)

# # # Take another cross product to get the third vector orthogonal to the first two
# # third_vector = np.cross(rotation_axis_normalized, cross_product_normalized)

# # # Construct the rotation matrix with these vectors as columns
# # rotation_matrix = np.column_stack((rotation_axis_normalized, cross_product_normalized, third_vector))

# # print("Rotation matrix:")
# # print(rotation_matrix)

# # # Convert the moon_position from the inertial frame to the rotating frame
# # moon_position_rotating = np.dot(rotation_matrix, moon_position)

# # print("Moon position in rotating frame:")
# # print(moon_position_rotating)

# # import matplotlib.pyplot as plt
# # # Plot the moon position vector
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.quiver(0, 0, 0, moon_position[0], moon_position[1], moon_position[2], color='blue', label='Moon Position')

# # # Plot the rotation axis vector
# # ax.quiver(0, 0, 0, rotation_axis[0], rotation_axis[1], rotation_axis[2], color='red', label='Rotation Axis')

# # # Set plot labels and legend
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')
# # ax.set_title('Moon Position and Rotation Axis')
# # ax.legend()

# # plt.show()

# # # Define the starting and ending keys
# # start_key = 60390
# # end_key = 60418

# # # Define the step size for each key range
# # skm_to_od_duration = 3
# # threshold = 3
# # od_duration = 1

# # # Generate the list of tuples representing the ranges of keys
# # key_ranges = []
# # last_key = start_key + threshold
# # for key in range(start_key, end_key, threshold):
# #     if key == start_key:
# #         key_ranges.append((key, key + threshold))
# #     else:
# #         key_ranges.append((last_key+skm_to_od_duration, last_key + skm_to_od_duration + od_duration))
# #         last_key += skm_to_od_duration + od_duration

# # print(key_ranges)



# import numpy as np

# # means = {}
# # # for num_run in range(100):

# # #     # Generate random variations
# # #     design_vector = np.random.normal(loc=1, scale=0.3, size=(1,10))
# # #     print(design_vector)
# # #     means[num_run] = design_vector

# # # print(means)


# # dynamic_model_list = ["HF", "PM", 0]
# # truth_model_list = ["HF", "PM", 0]
# # duration = 28
# # skm_to_od_duration = 3
# # threshold = 3
# # od_duration = 1
# # num_runs = 3
# # mean = od_duration
# # std_dev = 0.3
# # factors = [0.2, 0.5, 1.3]


# # np.random.seed(0)
# # for i in range(10):

# #     # Generate a vector with OD durations
# #     od_duration = np.random.normal(loc=1, scale=0.5, size=(20))
# #     start_epoch = 60390
# #     epoch = start_epoch + threshold + skm_to_od_duration + od_duration[0]
# #     skm_epochs = []
# #     i = 1
# #     while True:
# #         if epoch < start_epoch+duration:
# #             skm_epochs.append(epoch)
# #             epoch += skm_to_od_duration+od_duration[i]
# #         else:
# #             design_vector = np.ones(np.shape(skm_epochs))
# #             break
# #         i += 1

# #     # Extract observation windows
# #     observation_windows = [(start_epoch, start_epoch+threshold)]
# #     for i, skm_epoch in enumerate(skm_epochs):
# #         observation_windows.append((skm_epoch-od_duration[i], skm_epoch))

# import matplotlib.pyplot as plt
# import numpy as np

# data = {
#   "PM": {
#     "0": [
#       [
#         1.5292157037902991,
#         1.120047162510167,
#         1.2936213952317217,
#         1.6722679597604373,
#         1.5602673970449903
#       ],
#       3.881389078096338
#     ],
#     "1": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.629279735186108
#     ],
#     "2": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.6301105788313834
#     ],
#     "3": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.620106701960488
#     ],
#     "4": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.6272855136957003
#     ],
#     "5": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.6252055059217634
#     ],
#     "6": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.6212312688083137
#     ],
#     "7": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.621432219529683
#     ],
#     "8": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.624500887958655
#     ],
#     "9": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       3.620730602657655
#     ]
#   },
#   "PMSRP": {
#     "0": [
#       [
#         1.5292157037902991,
#         1.120047162510167,
#         1.2936213952317217,
#         1.6722679597604373,
#         1.5602673970449903
#       ],
#       0.30727913371425425
#     ],
#     "1": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.3032678512529212
#     ],
#     "2": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.3048521601479422
#     ],
#     "3": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.30142743380698
#     ],
#     "4": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.3005647548033069
#     ],
#     "5": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.29966621542917526
#     ],
#     "6": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.2979504188953162
#     ],
#     "7": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.30132144828734964
#     ],
#     "8": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.29843010496775163
#     ],
#     "9": [
#       [
#         1.6722679597604373,
#         1.5602673970449903,
#         0.7068166360370767,
#         1.2850265252576767,
#         0.9545928375106907
#       ],
#       0.3030935095215229
#     ]
#   }
# }

# # Extracting the second value from each subkey and calculating mean and standard deviation
# means_PM = []
# stds_PM = []
# means_PMSRP = []
# stds_PMSRP = []

# for i in range(10):
#     means_PM.append(np.mean([data["PM"][str(i)][1] for i in range(10)]))
#     stds_PM.append(np.std([data["PM"][str(i)][1] for i in range(10)]))
#     means_PMSRP.append(np.mean([data["PMSRP"][str(i)][1] for i in range(10)]))
#     stds_PMSRP.append(np.std([data["PMSRP"][str(i)][1] for i in range(10)]))

# # Creating the bar chart
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(2)

# bar1 = ax.bar(index, [means_PM[0], means_PMSRP[0]], bar_width, label='Iteration 0', yerr=[stds_PM[0], stds_PMSRP[0]])

# bar2 = ax.bar(index + bar_width, [means_PM[1], means_PMSRP[1]], bar_width, label='Iteration 1', yerr=[stds_PM[1], stds_PMSRP[1]])

# ax.set_xlabel('Keys')
# ax.set_ylabel('Values')
# ax.set_title('Mean and Standard Deviation of Each Key')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('PM', 'PMSRP'))
# ax.legend()

# plt.show()

import numpy as np

# Example state histories (replace these with your actual data)
state_history = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                          [2, 3, 4, 5, 6, 7, 8, 9, 10],
                          [3, 4, 5, 6, 7, 8, 9, 10, 11]])

dependent_variables_history = np.array([[1, 2, 3],
                                        [2, 3, 4],
                                        [3, 4, 5]])

# Calculate the vectors
states_history_LPF_moon = state_history[:, 0:3] - dependent_variables_history[:, 0:3]
states_history_LUMIO_moon = state_history[:, 6:9] - dependent_variables_history[:, 0:3]

# Calculate the dot product between the vectors
dot_product = np.dot(states_history_LPF_moon, states_history_LUMIO_moon)

print(dot_product)

# Calculate the magnitudes of the vectors
magnitude_LPF_moon = np.linalg.norm(states_history_LPF_moon, axis=1)
magnitude_LUMIO_moon = np.linalg.norm(states_history_LUMIO_moon, axis=1)

# Calculate the cosine of the angle
cosine_angle = dot_product / (magnitude_LPF_moon * magnitude_LUMIO_moon)

# Calculate the angle in radians
angle_rad = np.arccos(cosine_angle)

# Convert radians to degrees
angle_deg = np.degrees(angle_rad)

print("Angle between vectors in degrees:", angle_deg)
