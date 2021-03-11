# Performance Viewer

    This tool is used to visualize the tpu performance result on a html web page.

    The performance result is generated on a real tpu hardware.


## Usage:

#### 1. Run the model_runner with the --pmu out_file_name
   ``` model_runner --input model_in_fp32.npz --model model.cvimodel --output model_out.npz --pmu out_file_name ```

   which will generate two files: out_file_name_des.csv and out_file_name_layer.csv

#### 2. convert the pmu out_file_name_des.csv to model.json

  ``` python pmu2json.py --input out_file_name_des.csv --output model.json ```

#### 3. clone catapult
   ``` git clone https://github.com/catapult-project/catapult.git ```

#### 4. convert json to html
   ``` ./catapult/tracing/bin/trace2html model.json --output out.html```

#### 5. open out.html use chrome browser