# ee4740_project

Repository for the project in the course EE4740.

Make sure all the required packages are installed. An `environment.yml` is provided to install all the required packages. Note: CUDA is required for the running of this code.

The figures from the report can be recreated by going to the following files and looking at the comments:

- Figure 1:
    This figure was created by performing "test runs" and then saving the plot and putting them all in a Powerpoint presentation. So this figure cannot be recreated with code alone.
  - Files:
    - main_convex.py
    - main_biht.py
    - main_unnp.py

- Figure 2:
  - Files:
    - main_biht.py
  - Set the flags:
    - BIHT_TEST_S_LEVELS = True
    - PROCESS_DATA_BIHT_TEST_S_LEVELS = True
    - PLOT_SPARSITY_DISTRIBUTION_AND_NMSE_PSNR = True
  
- Figure 3:
  - Files:
    - main_unnp.py
  - Set the flags:
    - COMPARE_LOSS_DENOM_SQUARE = True

- Figure 4:
  - Files:
    - main_unnp.py
  - Set the flags:
    - UNNP_TEST_NUM_M = True
    - UNNP_TEST_NUM_M_LEAKYRELU = True
    - PROCESS_DATA_UNNP_TEST_NUM_M = True
    - PROCESS_DATA_UNNP_TEST_NUM_M_LEAKYRELU = True
    - PLOT_RESULTS_UNNP_TEST_NUM_M_LEAKYRELU_OLNY_NMSE = True

- Figure 5:
  - Files:
    - main_convex.py
    - main_biht.py
    - main_unnp.py
    - main.py (**run last**)
  - Set the flags:
    - CONVEX_RUN_M = True
    - BIHT_TEST_NUM_M = True
    - PROCESS_DATA_BIHT_TEST_NUM_M = True
    - UNNP_TEST_NUM_M = True
    - PROCESS_DATA_UNNP_TEST_NUM_M = True
    - FULL_COMPARISON_ALL = True
