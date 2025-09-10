import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "stanford_toddy": ASSET_ROOT / "stanford_toddy" / "toddy_mocap.xml",
    "stanford_toddy_vr": HERE / ".." / ".." / ".." / "descriptions" / "toddlerbot_2xc" / "scene_teleop.xml"
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "stanford_toddy": IK_CONFIG_ROOT / "smplx_to_toddy.json",
    },
    "bvh":{
        "stanford_toddy": IK_CONFIG_ROOT / "bvh_to_toddy.json",
    },
    "fbx":{
    },
    "fbx_offline":{
    },
    "human_quest":{
        "stanford_toddy_vr": HERE / ".." / "vr_configs" / "quest_toddy_gmr.yaml"
    }
}


ROBOT_BASE_DICT = {
    "stanford_toddy": "waist_link",
    "stanford_toddy_vr": "waist_gears",
}

VIEWER_CAM_DISTANCE_DICT = {
    "stanford_toddy": 1.0,
    "stanford_toddy_vr": 1.0,
}