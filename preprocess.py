import pandas as pd
import itertools

def preprocess_meta(df, df_miscut, ltt_path):
    STANDARD_MATERIAL = "S235"
    task_attributes =  ["thickness", "ltt_name", "machine_type", "adb", "laser_power"]
    
    df["versuchs_plan"] = df["edge_id"].str.split("-").str.get(1)

    df = df[
        [
            "edge_id",
            "thickness",
            "ltt_name",
            "laser_power",
            "machine_type",
            "adb",
            "material_type",
            "versuchs_plan",
            "burr_evaluated",
            "roughness_z_evaluated",
            "feedrate",
            "gas_pressure",
            "focal_position",
        ]
    ]
    df["material_type"] = df["material_type"].fillna(STANDARD_MATERIAL)

    df = df.dropna(axis=0)
    df.index = range(len(df.index))
    df.drop(columns=["edge_id", "versuchs_plan"], inplace=True)

    if df_miscut is not None:
        df_miscut = df_miscut[
            [
                "thickness",
                "ltt_name",
                "laser_power",
                "machine_type",
                "adb",
                "material_type",
                "feedrate",
                "gas_pressure",
                "focal_position",
            ]
        ]
        df_miscut["burr_evaluated"] = -1
        df_miscut["roughness_z_evaluated"] = -1
        df_miscut.loc[df_miscut["material_type"] == "Unknown", "material_type"] = (
            STANDARD_MATERIAL
        )
        df = pd.concat([df, df_miscut], ignore_index=True)

    df = (
        df.groupby(
            [
                "thickness",
                "ltt_name",
                "laser_power",
                "machine_type",
                "adb",
                "material_type",
                "feedrate",
                "gas_pressure",
                "focal_position",
            ]
        )[["burr_evaluated", "roughness_z_evaluated"]]
        .mean()
        .reset_index()
    )

    # df["target"] = df[["burr_evaluated", "roughness_z_evaluated"]].sum(axis=1)
    # remove gas pressures > 25 (most likely false data)
    df = df[df["gas_pressure"] <= 25]

    combination_list = list(
        itertools.product(*[df[col].unique() for col in task_attributes])
    )
    unique_tasks_expected_strings = []
    for combination in combination_list:
        unique_tasks_expected_strings.append(
            "_".join([str(x) for x in combination])
        )

    unique_materials = df["material_type"].unique().tolist()

    unique_combinations = pd.read_csv(
        ltt_path,
        dtype={
            "thickness": "int",
            "ltt_name": "string",
            "machine_type": "string",
            "adb": "float",
            "laser_power": "int",
            "material_type": "string",
        },
    )
    unique_combinations["combination_string"] = unique_combinations.apply(
        lambda x: "_".join([str(x[elem]) for elem in task_attributes]), axis=1
    )
    unique_combinations = unique_combinations[
        unique_combinations["combination_string"].isin(
            unique_tasks_expected_strings
        )
    ].reset_index(drop=True)

    unique_combinations = unique_combinations.sort_values(
        by="combination_string"
    ).reset_index(drop=True)

    task_df_dict = _get_df_per_task(df, materials=unique_materials, unique_combinations=unique_combinations, task_attributes=task_attributes)

    task_df_dict = _remove_data_duplicates(task_df_dict)
    
    task_df_dict = _remove_tasks_without_enough_data(task_df_dict)

    return task_df_dict, unique_combinations

def _get_df_per_task(df_origin, materials, unique_combinations, task_attributes):
        task_df_dict = {}
        for i, row in unique_combinations.iterrows():
            str_combination = row["combination_string"]
            df = df_origin.copy()

            for elem in task_attributes:
                df = df[df[elem] == row[elem]]

            for material in materials:
                df_material = df[df["material_type"] == material]
                if not df_material.empty and not len(df_material.index) == 1:
                    str_combination_material = str_combination + "_" + material

                    task_df_dict[str_combination_material] = df_material
        return task_df_dict
    
def _remove_tasks_without_enough_data(
        task_df_dict,
    ):
        keys_to_remove = []
        for key, df in task_df_dict.items():
            if len(df.index) < 3:
                keys_to_remove.append(key)
        for elem in keys_to_remove:
            del task_df_dict[elem]
        return task_df_dict

def _remove_data_duplicates(task_df_dict):
    for key in task_df_dict:
        task_df_dict[key] = task_df_dict[key].drop_duplicates(
            subset=["feedrate", "gas_pressure", "focal_position"]
        )
    return task_df_dict