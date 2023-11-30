
#!/bin/bash

# Read the case.json file
case_file="case.json"

bc=$(jq -r '.Boundary_condition' "$case_file")
pde=$(jq -r '.Class_PDE' "$case_file")
geom=$(jq -r '.Geometry' "$case_file")
num_pb=$(jq -r '.Problem' "$case_file")
sampling=$(jq -r '.Sampling_on' "$case_file")

# Create the name of the directory
geom_name="networks/$bc/$pde/$geom"
sol='_Solution'
train='_training'
solname="$geom_name$sol$num_pb"
dir_name="$variablename$solname/$sampling$train"
# echo "$dir_name"

# Search for all config_i files in dir_name
config_files=$(find "$dir_name/models" -name "config_*")
echo "$config_files"

# Extract the part between "config_" and ".json"
num_config=$(basename -a $config_files | sed 's/config_\(.*\)\.json/\1/')
# echo "$num_config"

# Run the models
for i in $num_config
do
    echo "### Apply correction for model $i"
    python3 run_corr.py --config $i
done