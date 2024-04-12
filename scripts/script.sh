# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_bean_1 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_bean_2 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_bean_3 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_bean_3 --config 2

# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_circle_1 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_circle_2 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_circle_3 --config 1
# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_circle_3 --config 2

# python3 pred_vs_ref_levelset_mesh.py --casefile_name lap_pumpkin_1 --config 1

python3 learn_Poisson2D.py --casefile lap_bean_1.json --n_layers 6 --units 64 --lr 0.007
python3 learn_Poisson2D.py --casefile lap_bean_2.json --n_layers 6 --units 64 --lr 0.007
python3 learn_Poisson2D.py --casefile lap_circle_1.json --n_layers 6 --units 64 --lr 0.007
python3 learn_Poisson2D.py --casefile lap_pumpkin_1.json --n_layers 6 --units 64 --lr 0.007