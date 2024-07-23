# Instructions for Evaluation

## Evaluation Script
The evaluation of LLMServingSim consists of five parts in total. 
Each evaluation script includes the `main.py` command according to the configuration of each evaluation. 
The output of these scripts is stored in their respective evaluation folders. 
As introduced in the LLMServingSim `README`, there are three outputs for each command, including the standard output redirected to a txt file.

### Run Each Evaluation Script
Should run these scripts in `evaluation` directory.

```bash
./evaluation1.sh
./evaluation2.sh
...
./evaluation5.sh
```

### Run All Evaluation Script

```bash
./evaluation_all.sh
```

## Using Excel File (Highly Recommended)
To facilitate verification of the results used in the paper, an Excel file `evaluation.xlsx` has been provided. 
Each Excel sheet represents a different evaluation, and they contain the numbers used in the paper. 
Additionally, explanations are included on how each number was obtained from the raw output.
The file also enables easy reproduction of figures by automatically generating them when the experimental results are pasted.

## Not Using Excel File
For those who find using Excel difficult, the results of the evaluation are also provided as CSV files in the `numbers_in_paper` folder.
However, since the Excel file includes instructions on how to use the raw data and easily create figures, using Excel is highly recommended.

## Archived Results
The results of running the same evaluation script on the hardware used in Section 6.1 are included in the `numbers_archived` folder. 
Please refer to this if needed.

## Units of Results and Figure
In each evaluation folder, `README` file indicates the unit of the result and figure.

## Important Notes
Note that simulation time can vary depending on hardware specifications. For details on the hardware we used, please refer to the Section 6.1
