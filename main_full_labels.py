"""
    This file shows how to load and use the dataset
"""

import json
import os

import numpy as np
import warnings

from PIL import Image
import multiprocessing as mp
import pycococreatortools
import datetime

FULL_CPU = True

INFO = {
    "description": "Mapillary",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "Luodian",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]

CATEGORIES = [
    {
        "id": 1,
        "name": "Bird",
        "supercategory": "animal--bird",
        "color": [
              165,
              42,
              42
          ]
    },
    {
        "id": 2,
        "name": "Ground_Animal",
        "supercategory": "animal--ground-animal",
        "color": [
              0,
              192,
              0
          ]
    },
    {
        "id": 3,
        "supercategory": "construction--barrier--ambiguous",
        "name": "Ambiguous_Barrier",
        "color": [
              250,
              170,
              31
          ]
    },
    {
        "id": 4,
        "supercategory": "construction--barrier--concrete-block",
        "name": "Concrete_Block",
        "color": [
              250,
              170,
              32
          ]
    },
    {
        "id": 5,
        "supercategory": "construction--barrier--curb",
        "name": "Curb",
        "color": [
                196,
                196,
                196
            ]
    },
    {
        "id": 6,
        "supercategory": "construction--barrier--fence",
        "name": "Fence",
        "color": [
                190,
                153,
                153
            ]
    },
    {
        "id": 7,
        "supercategory": "construction--barrier--guard-rail",
        "name": "Guard_Rail",
        "color": [
                180,
                165,
                180
            ]
    },
    {
        "id": 8,
        "supercategory": "construction--barrier--other-barrier",
        "name": "Barrier",
        "color": [
                90,
                120,
                150
            ]
    },
    {
        "id": 9,
        "supercategory": "construction--barrier--road-median",
        "name": "Road_Median",
        "color": [
                90,
                120,
                150
            ]
    },
    {
        "id": 10,
        "supercategory": "construction--barrier--road-side",
        "name": "Road_Side",
        "color": [
                250,
                170,
                34
            ]
    },
    {
        "id": 11,
        "supercategory": "construction--barrier--separator",
        "name": "Lane_Separator",
        "color": [
                128,
                128,
                128
            ]
    },
    {
        "id": 12,
        "supercategory": "construction--barrier--temporary",
        "name": "Temporary_Barrier",
        "color": [
                250,
                170,
                35
            ]
    },
    {
        "id": 13,
        "supercategory": "construction--barrier--wall",
        "name": "Wall",
        "color": [
                102,
                102,
                156
            ]
    },
    {
        "id": 14,
        "supercategory": "construction--flat--bike-lane",
        "name": "Bike_Lane",
        "color": [
                128,
                64,
                255
            ]
    },
    {
        "id": 15,
        "name": "Crosswalk_Plain",
        "supercategory": "construction--flat--crosswalk-plain",
        "color": [
                140,
                140,
                200
            ]
    },
    {
        "id": 16,
        "supercategory": "construction--flat--curb-cut",
        "name": "Curb_Cut",
        "color": [
                170,
                170,
                170
            ]
    },
    {
        "id": 17,
        "supercategory": "construction--flat--driveway",
        "name": "Driveway",
        "color": [
                250,
                170,
                36
            ]
    },
    {
        "id": 18,
        "supercategory": "construction--flat--parking",
        "name": "Parking",
        "color": [
                250,
                170,
                160
            ]
    },
    {
        "id": 19,
        "supercategory": "construction--flat--parking-aisle",
        "name": "Parking_Aisle",
        "color": [
                250,
                170,
                37
            ]
    },
    {
        "id": 20,
        "supercategory": "construction--flat--pedestrian-area",
        "name": "Pedestrian_Area",
        "color": [
                96,
                96,
                96
            ]
    },
    {
        "id": 21,
        "supercategory": "construction--flat--rail-track",
        "name": "Rail_Track",
        "color": [
                230,
                150,
                140
            ]
    },
    {
        "id": 22,
        "supercategory": "construction--flat--road",
        "name": "Road",
        "color": [
                128,
                64,
                128
            ]
    },
    {
        "id": 23,
        "supercategory": "construction--flat--road-shoulder",
        "name": "Road_Shoulder",
        "color": [
                110,
                110,
                110
            ]
    },
    {
        "id": 24,
        "supercategory": "construction--flat--service-lane",
        "name": "Service_Lane",
        "color": [
                110,
                110,
                110
            ]
    },
    {
        "id": 25,
        "supercategory": "construction--flat--sidewalk",
        "name": "Sidewalk",
        "color": [
                244,
                35,
                232
            ]
    },
    {
        "id": 26,
        "supercategory": "construction--flat--traffic-island",
        "name": "Traffic_Island",
        "color": [
                128,
                196,
                128
            ]
    },
    {
        "id": 27,
        "supercategory": "construction--structure--bridge",
        "name": "Bridge",
        "color": [
                150,
                100,
                100
            ]
    },
    {
        "id": 28,
        "supercategory": "construction--structure--building",
        "name": "Building",
        "color": [
                70,
                70,
                70
            ]
    },
    {
        "id": 29,
        "supercategory": "construction--structure--garage",
        "name": "Garage",
        "color": [
                150,
                150,
                150
            ]
    },
    {
        "id": 30,
        "supercategory": "construction--structure--tunnel",
        "name": "Tunnel",
        "color": [
                150,
                120,
                90
            ]
    },
    {
        "id": 31,
        "name": "Person",
        "supercategory": "human--person--individual",
        "color": [
                220,
                20,
                60
            ]
    },
    {
        "id": 32,
        "supercategory": "human--person--person-group",
        "name": "Person_Group",
        "color": [
                220,
                20,
                60
            ]
    },
    {
        "id": 33,
        "name": "Bicyclist",
        "supercategory": "human--rider--bicyclist",
        "color": [
                255,
                0,
                0
            ]
    },
    {
        "id": 34,
        "name": "Motorcyclist",
        "supercategory": "human--rider--motorcyclist",
        "color": [
                255,
                0,
                100
            ]
    },
    {
        "id": 35,
        "name": "Other_Rider",
        "supercategory": "human--rider--other-rider",
        "color": [
                255,
                0,
                200
            ]
    },
    {
        "id": 36,
        "name": "Lane_Marking_-_Dashed_Line",
        "supercategory": "marking--continuous--dashed",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 37,
        "name": "Lane_Marking_-_Straight_Line",
        "supercategory": "marking--continuous--solid",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 38,
        "name": "Lane_Marking_-_Zigzag_Line",
        "supercategory": "marking--continuous--zigzag",
        "color": [
                250,
                170,
                29
            ]
    },
    {
        "id": 39,
        "name": "Lane_Marking_-_Ambiguous",
        "supercategory": "marking--discrete--ambiguous",
        "color": [
                250,
                170,
                28
            ]
    },
    {
        "id": 40,
        "name": "Lane_Marking_-_Arrow_(Left)",
        "supercategory": "marking--discrete--arrow--left",
        "color": [
                250,
                170,
                26
            ]
    },
    {
        "id": 41,
        "name": "Lane_Marking_-_Arrow_(Other)",
        "supercategory": "marking--discrete--arrow--other",
        "color": [
                250,
                170,
                25
            ]
    },
    {
        "id": 42,
        "name": "Lane_Marking_-_Arrow_(Right)",
        "supercategory": "marking--discrete--arrow--right",
        "color": [
                250,
                170,
                24
            ]
    },
    {
        "id": 43,
        "name": "Lane_Marking_-_Arrow_(Split_Left_or_Straight)",
        "supercategory": "marking--discrete--arrow--split-left-or-straight",
        "color": [
                250,
                170,
                22
            ]
    },
    {
        "id": 44,
        "name": "Lane_Marking_-_Arrow_(Split_Right_or_Straight)",
        "supercategory": "marking--discrete--arrow--split-right-or-straight",
        "color": [
                250,
                170,
                21
            ]
    },
    {
        "id": 45,
        "name": "Lane_Marking_-_Arrow_(Straight)",
        "supercategory": "marking--discrete--arrow--straight",
        "color": [
                250,
                170,
                20
            ]
    },
    {
        "id": 46,
        "name": "Lane_Marking_-_Crosswalk",
        "supercategory": "marking--discrete--crosswalk-zebra",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 47,
        "supercategory": "marking--discrete--give-way-row",
        "name": "Lane_Marking_-_Give_Way_(Row)",
        "color": [
                250,
                170,
                19
            ]
    },
    {
        "id": 48,
        "supercategory": "marking--discrete--give-way-single",
        "name": "Lane_Marking_-_Give_Way_(Single)",
        "color": [
                250,
                170,
                18
            ]
    },
    {
        "id": 49,
        "supercategory": "marking--discrete--hatched--chevron",
        "name": "Lane_Marking_-_Hatched_(Chevron)",
        "color": [
                250,
                170,
                12
            ]
    },
    {
        "id": 50,
        "supercategory": "marking--discrete--hatched--diagonal",
        "name": "Lane_Marking_-_Hatched_(Diagonal)",
        "color": [
                250,
                170,
                11
            ]
    },
    {
        "id": 51,
        "supercategory": "marking--discrete--other-marking",
        "name": "Lane_Marking_-_Other",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 52,
        "supercategory": "marking--discrete--stop-line",
        "name": "Lane_Marking_-_Stop_Line",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 53,
        "supercategory": "marking--discrete--symbol--bicycle",
        "name": "Lane_Marking_-_Symbol_(Bicycle)",
        "color": [
                250,
                170,
                16
            ]
    },
    {
        "id": 54,
        "supercategory": "marking--discrete--symbol--other",
        "name": "Lane_Marking_-_Symbol_(Other)",
        "color": [
                250,
                170,
                15
            ]
    },
    {
        "id": 55,
        "supercategory": "marking--discrete--text",
        "name": "Lane_Marking_-_Text",
        "color": [
                250,
                170,
                15
            ]
    },
    {
        "id": 56,
        "supercategory": "marking-only--continuous--dashed",
        "name": "Lane_Marking_(only)_-_Dashed_Line",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 57,
        "supercategory": "marking-only--discrete--crosswalk-zebra",
        "name": "Lane_Marking_(only)_-_Crosswalk",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 58,
        "supercategory": "marking-only--discrete--other-marking",
        "name": "Lane_Marking_(only)_-_Other",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 59,
        "supercategory": "marking-only--discrete--text",
        "name": "Lane_Marking_(only)_-_Test",
        "color": [
                255,
                255,
                255
            ]
    },
    {
        "id": 60,
        "supercategory": "nature--mountain",
        "name": "Mountain",
        "color": [
                64,
                170,
                64
            ]
    },
    {
        "id": 61,
        "supercategory": "nature--sand",
        "name": "Sand",
        "color": [
                230,
                160,
                50
            ]
    },
    {
        "id": 62,
        "supercategory": "nature--sky",
        "name": "Sky",
        "color": [
                70,
                130,
                180
            ]
    },
    {
        "id": 63,
        "supercategory": "nature--snow",
        "name": "Snow",
        "color": [
                190,
                255,
                255
            ]
    },
    {
        "id": 64,
        "supercategory": "nature--terrain",
        "name": "Terrain",
        "color": [
                152,
                251,
                152
            ]
    },
    {
        "id": 65,
        "supercategory": "nature--vegetation",
        "name": "Vegetation",
        "color": [
                107,
                142,
                35
            ]
    },
    {
        "id": 66,
        "supercategory": "nature--water",
        "name": "Water",
        "color": [
                0,
                170,
                30
            ]
    },
    {
        "id": 67,
        "name": "Banner",
        "supercategory": "object--banner",
        "color": [
                255,
                255,
                128
            ]
    },
    {
        "id": 68,
        "name": "Bench",
        "supercategory": "object--bench",
        "color": [
                250,
                0,
                30
            ]
    },
    {
        "id": 69,
        "name": "Bike_Rack",
        "supercategory": "object--bike-rack",
        "color": [
                100,
                140,
                180
            ]
    },
    {
        "id": 70,
        "name": "Catch_Basin",
        "supercategory": "object--catch-basin",
        "color": [
                220,
                128,
                128
            ]
    },
    {
        "id": 71,
        "name": "CCTV_Camera",
        "supercategory": "object--cctv-camera",
        "color": [
                222,
                40,
                40
            ]
    },
    {
        "id": 72,
        "name": "Fire_Hydrant",
        "supercategory": "object--fire-hydrant",
        "color": [
                100,
                170,
                30
            ]
    },
    {
        "id": 73,
        "name": "Junction_Box",
        "supercategory": "object--junction-box",
        "color": [
                40,
                40,
                40
            ]
    },
    {
        "id": 74,
        "name": "Mailbox",
        "supercategory": "object--mailbox",
        "color": [
                33,
                33,
                33
            ]
    },
    {
        "id": 75,
        "name": "Manhole",
        "supercategory": "object--manhole",
        "color": [
                100,
                128,
                160
            ]
    },
    {
        "id": 76,
        "name": "Parking_Meter",
        "supercategory": "object--parking-meter",
        "color": [
                20,
                20,
                255
            ]
    },
    {
        "id": 77,
        "name": "Phone_Booth",
        "supercategory": "object--phone-booth",
        "color": [
                142,
                0,
                0
            ]
    },
    {
        "id": 78,
        "supercategory": "object--pothole",
        "name": "Pothole",
        "color": [
                70,
                100,
                150
            ]
    },
    {
        "id": 79,
        "supercategory": "object--sign--advertisement",
        "name": "Signage_-_Advertisement",
        "color": [
                250,
                171,
                30
            ]
    },
    {
        "id": 80,
        "supercategory": "object--sign--ambiguous",
        "name": "Signage_-_Ambiguous",
        "color": [
                250,
                172,
                30
            ]
    },
    {
        "id": 81,
        "supercategory": "object--sign--back",
        "name": "Signage_-_Back",
        "color": [
                250,
                173,
                30
            ]
    },
    {
        "id": 82,
        "supercategory": "object--sign--information",
        "name": "Signage_-_Information",
        "color": [
                250,
                174,
                30
            ]
    },
    {
        "id": 83,
        "supercategory": "object--sign--other",
        "name": "Signage_-_Other",
        "color": [
            250,
            175,
            30
        ]
    },
    {
        "id": 84,
        "supercategory": "object--sign--store",
        "name": "Signage_-_Store",
        "color": [
                250,
                176,
                30
            ]
    },
    {
        "id": 85,
        "name": "Street_Light",
        "supercategory": "object--street-light",
        "color": [
                210,
                170,
                100
            ]
    },
    {
        "id": 86,
        "name": "Pole",
        "supercategory": "object--support--pole",
        "color": [
                153,
                153,
                153
            ]
    },
    {
        "id": 87,
        "name": "Pole_Group",
        "supercategory": "object--support--pole-group",
        "color": [
                153,
                153,
                153
            ]
    },
    {
        "id": 88,
        "name": "Traffic_Sign_Frame",
        "supercategory": "object--support--traffic-sign-frame",
        "color": [
                128,
                128,
                128
            ]
    },
    {
        "id": 89,
        "name": "Utility_Pole",
        "supercategory": "object--support--utility-pole",
        "color": [
                0,
                0,
                80
            ]
    },
    {
        "id": 90,
        "supercategory": "object--traffic-cone",
        "name": "Traffic_Cone",
        "color": [
                210,
                60,
                60
            ]
    },
    {
        "id": 91,
        "supercategory": "object--traffic-light--general-single",
        "name": "Traffic_Light_-_General_(Single)",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 92,
        "supercategory": "object--traffic-light--pedestrians",
        "name": "Traffic_Light_-_Pedestrians",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 93,
        "supercategory": "object--traffic-light--general-upright",
        "name": "Traffic_Light_-_General_(Upright)",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 94,
        "supercategory": "object--traffic-light--general-horizontal",
        "name": "Traffic_Light_-_General_(Horizontal)",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 95,
        "supercategory": "object--traffic-light--cyclists",
        "name": "Traffic_Light_-_Cyclists",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 96,
        "supercategory": "object--traffic-light--other",
        "name": "Traffic_Light_-_Other",
        "color": [
                250,
                170,
                30
            ]
    },
    {
        "id": 97,
        "supercategory": "object--traffic-sign--ambiguous",
        "name": "Traffic_Sign_-_Ambiguous",
        "color": [
                192,
                192,
                192
            ]
    },
    {
        "id": 98,
        "name": "Traffic_Sign_(Back)",
        "supercategory": "object--traffic-sign--back",
        "color": [
                192,
                192,
                192
            ]
    },
    {
        "id": 99,
        "supercategory": "object--traffic-sign--direction-back",
        "name": "Traffic_Sign_-_Direction_(Back)",
        "color": [
                192,
                192,
                192
            ]
    },
    {
        "id": 100,
        "supercategory": "object--traffic-sign--direction-front",
        "name": "Traffic_Sign_-_Direction_(Front)",
        "color": [
                220,
                220,
                0
            ]
    },
    {
        "id": 101,
        "name": "Traffic_Sign_(Front)",
        "supercategory": "object--traffic-sign--front",
        "color": [
                220,
                220,
                0
            ]
    },
    {
        "id": 102,
        "supercategory": "object--traffic-sign--information-parking",
        "name": "Traffic_Sign_-_Parking",
        "color": [
                0,
                0,
                196
            ]
    },
    {
        "id": 103,
        "supercategory": "object--traffic-sign--temporary-back",
        "name": "Traffic_Sign_-_Temporary_(Back)",
        "color": [
                192,
                192,
                192
            ]
    },
    {
        "id": 104,
        "supercategory": "object--traffic-sign--temporary-front",
        "name": "Traffic_Sign_-_Temporary_(Front)",
        "color": [
                220,
                220,
                0
            ]
    },
    {
        "id": 105,
        "name": "Trash_Can",
        "supercategory": "object--trash-can",
        "color": [
                140,
                140,
                20
            ]
    },
    {
        "id": 106,
        "name": "Bicycle",
        "supercategory": "object--vehicle--bicycle",
        "color": [
                119,
                11,
                32
            ]
    },
    {
        "id": 107,
        "name": "Boat",
        "supercategory": "object--vehicle--boat",
        "color": [
                150,
                0,
                255
            ]
    },
    {
        "id": 108,
        "name": "Bus",
        "supercategory": "object--vehicle--bus",
        "color": [
                0,
                60,
                100
            ]
    },
    {
        "id": 109,
        "name": "Car",
        "supercategory": "object--vehicle--car",
        "color": [
                0,
                0,
                142
            ]
    },
    {
        "id": 110,
        "name": "Caravan",
        "supercategory": "object--vehicle--caravan",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 111,
        "name": "Motorcycle",
        "supercategory": "object--vehicle--motorcycle",
        "color": [
                0,
                0,
                230
            ]
    },
    {
        "id": 112,
        "name": "On_Rails",
        "supercategory": "object--vehicle--on-rails",
        "color": [
                0,
                80,
                100
            ]
    },
    {
        "id": 113,
        "name": "Other_Vehicle",
        "supercategory": "object--vehicle--other-vehicle",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 114,
        "name": "Trailer",
        "supercategory": "object--vehicle--trailer",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 115,
        "name": "Truck",
        "supercategory": "object--vehicle--truck",
        "color": [
                0,
                0,
                70
            ]
    },
    {
        "id": 116,
        "supercategory": "object--vehicle--vehicle-group",
        "name": "Vehicle_Group",
        "color": [
                0,
                0,
                142
            ]
    },
    {
        "id": 117,
        "name": "Wheeled_Slow",
        "supercategory": "object--vehicle--wheeled-slow",
        "color": [
                0,
                0,
                192
            ]
    },
    {
        "id": 118,
        "supercategory": "object--water-valve",
        "name": "Water_Valve",
        "color": [
                170,
                170,
                170
            ]
    },
    {
        "id": 119,
        "supercategory": "void--car-mount",
        "name": "Car_Mount",
        "color": [
                32,
                32,
                32
            ]
    },
    {
        "id": 120,
        "supercategory": "void--dynamic",
        "name": "Dynamic",
        "color": [
                111,
                74,
                0
            ]
    },
    {
        "id": 121,
        "supercategory": "void--ego-vehicle",
        "name": "Ego_Vehicle",
        "color": [
                120,
                10,
                10
            ]
    },
    {
        "id": 122,
        "supercategory": "void--ground",
        "name": "Ground",
        "color": [
                81,
                0,
                81
            ]
    },
    {
        "id": 123,
        "supercategory": "void--static",
        "name": "Static",
        "color": [
                111,
                111,
                0
            ]
    },
    {
        "id": 124,
        "supercategory": "void--unlabeled",
        "name": "Unlabeled",
        "color": [
                0,
                0,
                0
            ]
    }
    
]


def split_to_coco_creator(input_instance_array, labels):
    labelid_matrix_name = []
    
    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)
        
    instance_image_info = np.array(input_instance_array % 256, dtype=np.uint8)

    unique_label_info = np.unique(label_image_info)
    unique_instance_info = np.unique(instance_image_info)

    for label_id, label in enumerate(labels):
        if (label_id in (unique_label_info)) and (label["instances"] == True):
            each_label_array = np.zeros(
                (input_instance_array.shape[0], input_instance_array.shape[1]),
                dtype=np.uint8,
            )

            each_label_array[label_image_info == label_id] = 255

            for instance_id in range(256):
                if instance_id in unique_instance_info:
                    each_instance_array = np.zeros(
                        (input_instance_array.shape[0],
                         input_instance_array.shape[1]),
                        dtype=np.uint8,
                    )

                    each_instance_array[instance_image_info ==
                                        instance_id] = 255

                    final_instance_array = np.bitwise_and(
                        each_instance_array, each_label_array
                    )

                    if np.unique(final_instance_array).size == 2:
                        labelid_matrix_name.append(
                            {
                                "label_id": label_id,
                                "instance_id": instance_id,
                                "label_name": label["readable"],
                                "image": final_instance_array,
                            }
                        )

    return labelid_matrix_name


def convert_class_id(annotation_filename):
    class_id = 0
    if "Bird" == annotation_filename:
        class_id = 1
    elif "Ground Animal" == annotation_filename:
        class_id = 2
    elif "Ambiguous Barrier" == annotation_filename:
        class_id = 3
    elif "Concrete Block" == annotation_filename:
        class_id = 4
    elif "Curb" == annotation_filename:
        class_id = 5
    elif "Fence" == annotation_filename:
        class_id = 6
    elif "Guard Rail" == annotation_filename:
        class_id = 7
    elif "Barrier" == annotation_filename:
        class_id = 8
    elif "Road Median" == annotation_filename:
        class_id = 9
    elif "Road Side" == annotation_filename:
        class_id = 10
    elif "Lane Separator" == annotation_filename:
        class_id = 11
    elif "Temporary Barrier" == annotation_filename:
        class_id = 12
    elif "Wall" == annotation_filename:
        class_id = 13
    elif "Bike Lane" == annotation_filename:
        class_id = 14
    elif "Crosswalk - Plain" == annotation_filename:
        class_id = 15
    elif "Curb Cut" == annotation_filename:
        class_id = 16
    elif "Driveway" == annotation_filename:
        class_id = 17
    elif "Parking" == annotation_filename:
        class_id = 18
    elif "Parking Aisle" == annotation_filename:
        class_id = 19
    elif "Pedestrian Area" == annotation_filename:
        class_id = 20
    elif "Rail Track" == annotation_filename:
        class_id = 21
    elif "Road" == annotation_filename:
        class_id = 22
    elif "Road Shoulder" == annotation_filename:
        class_id = 23
    elif "Service Lane" == annotation_filename:
        class_id = 24
    elif "Sidewalk" == annotation_filename:
        class_id = 25
    elif "Traffic Island" == annotation_filename:
        class_id = 26
    elif "Bridge" == annotation_filename:
        class_id = 27
    elif "Building" == annotation_filename:
        class_id = 28
    elif "Garage" == annotation_filename:
        class_id = 29
    elif "Tunnel" == annotation_filename:
        class_id = 30
    elif "Person" == annotation_filename:
        class_id = 31
    elif "Person Group" == annotation_filename:
        class_id = 32
    elif "Bicyclist" == annotation_filename:
        class_id = 33
    elif "Motorcyclist" == annotation_filename:
        class_id = 34
    elif "Other Rider" == annotation_filename:
        class_id = 35
    elif "Lane Marking - Dashed Line" == annotation_filename:
        class_id = 36
    elif "Lane Marking - Straight Line" == annotation_filename:
        class_id = 37
    elif "Lane Marking - Zigzag Line" == annotation_filename:
        class_id = 38
    elif "Lane Marking - Ambiguous" == annotation_filename:
        class_id = 39
    elif "Lane Marking - Arrow (Left)" == annotation_filename:
        class_id = 40
    elif "Lane Marking - Arrow (Other)" == annotation_filename:
        class_id = 41
    elif "Lane Marking - Arrow (Right)" == annotation_filename:
        class_id = 42
    elif "Lane Marking - Arrow (Split Left or Straight)" == annotation_filename:
        class_id = 43
    elif "Lane Marking - Arrow (Split Right or Straight)" == annotation_filename:
        class_id = 44
    elif "Lane Marking - Arrow (Straight)" == annotation_filename:
        class_id = 45
    elif "Lane Marking - Crosswalk" == annotation_filename:
        class_id = 46
    elif "Lane Marking - Give Way (Row)" == annotation_filename:
        class_id = 47
    elif "Lane Marking - Give Way (Single)" == annotation_filename:
        class_id = 48
    elif "Lane Marking - Hatched (Chevron)" == annotation_filename:
        class_id = 49
    elif "Lane Marking - Hatched (Diagonal)" == annotation_filename:
        class_id = 50
    elif "Lane Marking - Other" == annotation_filename:
        class_id = 51
    elif "Lane Marking - Stop Line" == annotation_filename:
        class_id = 52
    elif "Lane Marking - Symbol (Bicycle)" == annotation_filename:
        class_id = 53
    elif "Lane Marking - Symbol (Other)" == annotation_filename:
        class_id = 54
    elif "Lane Marking - Text" == annotation_filename:
        class_id = 55
    elif "Lane Marking (only) - Dashed Line" == annotation_filename:
        class_id = 56
    elif "Lane Marking (only) - Crosswalk" == annotation_filename:
        class_id = 57
    elif "Lane Marking (only) - Other" == annotation_filename:
        class_id = 58
    elif "Lane Marking (only) - Test" == annotation_filename:
        class_id = 59
    elif "Mountain" == annotation_filename:
        class_id = 60
    elif "Sand" == annotation_filename:
        class_id = 61
    elif "Sky" == annotation_filename:
        class_id = 62
    elif "Snow" == annotation_filename:
        class_id = 63
    elif "Terrain" == annotation_filename:
        class_id = 64
    elif "Vegetation" == annotation_filename:
        class_id = 65
    elif "Water" == annotation_filename:
        class_id = 66
    elif "Banner" == annotation_filename:
        class_id = 67
    elif "Bench" == annotation_filename:
        class_id = 68
    elif "Bike Rack" == annotation_filename:
        class_id = 69
    elif "Catch Basin" == annotation_filename:
        class_id = 70
    elif "CCTV Camera" == annotation_filename:
        class_id = 71
    elif "Fire Hydrant" == annotation_filename:
        class_id = 72
    elif "Junction Box" == annotation_filename:
        class_id = 73
    elif "Mailbox" == annotation_filename:
        class_id = 74
    elif "Manhole" == annotation_filename:
        class_id = 75
    elif "Parking Meter" == annotation_filename:
        class_id = 76
    elif "Phone Booth" == annotation_filename:
        class_id = 77
    elif "Pothole" == annotation_filename:
        class_id = 78
    elif "Signage - Advertisement" == annotation_filename:
        class_id = 79
    elif "Signage - Ambiguous" == annotation_filename:
        class_id = 80
    elif "Signage - Back" == annotation_filename:
        class_id = 81
    elif "Signage - Information" == annotation_filename:
        class_id = 82
    elif "Signage - Other" == annotation_filename:
        class_id = 83
    elif "Signage - Store" == annotation_filename:
        class_id = 84
    elif "Street Light" == annotation_filename:
        class_id = 85
    elif "Pole" == annotation_filename:
        class_id = 86
    elif "Pole Group" == annotation_filename:
        class_id = 87
    elif "Traffic Sign Frame" == annotation_filename:
        class_id = 88
    elif "Utility Pole" == annotation_filename:
        class_id = 89
    elif "Traffic Cone" == annotation_filename:
        class_id = 90
    elif "Traffic Light - General (Single)" == annotation_filename:
        class_id = 91
    elif "Traffic Light - Pedestrians" == annotation_filename:
        class_id = 92
    elif "Traffic Light - General (Upright)" == annotation_filename:
        class_id = 93
    elif "Traffic Light - General (Horizontal)" == annotation_filename:
        class_id = 94
    elif "Traffic Light - Cyclists" == annotation_filename:
        class_id = 95
    elif "Traffic Light - Other" == annotation_filename:
        class_id = 96
    elif "Traffic Sign - Ambiguous" == annotation_filename:
        class_id = 97
    elif "Traffic Sign (Back)" == annotation_filename:
        class_id = 98
    elif "Traffic Sign - Direction (Back)" == annotation_filename:
        class_id = 99
    elif "Traffic Sign - Direction (Front)" == annotation_filename:
        class_id = 100
    elif "Traffic Sign (Front)" == annotation_filename:
        class_id = 101
    elif "Traffic Sign - Parking" == annotation_filename:
        class_id = 102
    elif "Traffic Sign - Temporary (Back)" == annotation_filename:
        class_id = 103
    elif "Traffic Sign - Temporary (Front)" == annotation_filename:
        class_id = 104
    elif "Trash Can" == annotation_filename:
        class_id = 105
    elif "Bicycle" == annotation_filename:
        class_id = 106
    elif "Boat" == annotation_filename:
        class_id = 107
    elif "Bus" == annotation_filename:
        class_id = 108
    elif "Car" == annotation_filename:
        class_id = 109
    elif "Caravan" == annotation_filename:
        class_id = 110
    elif "Motorcycle" == annotation_filename:
        class_id = 111
    elif "On Rails" == annotation_filename:# elif "Unlabeled" == annotation_filename:
        class_id = 124
        class_id = 112
    elif "Other Vehicle" == annotation_filename:
        class_id = 113
    elif "Trailer" == annotation_filename:
        class_id = 114
    elif "Truck" == annotation_filename:
        class_id = 115
    elif "Vehicle Group" == annotation_filename:
        class_id = 116
    elif "Wheeled Slow" == annotation_filename:
        class_id = 117
    elif "Water Valve" == annotation_filename:
        class_id = 118
    elif "Car Mount" == annotation_filename:
        class_id = 119
    elif "Dynamic" == annotation_filename:
        class_id = 120
    elif "Ego Vehicle" == annotation_filename:
        class_id = 121
    elif "Ground" == annotation_filename:
        class_id = 122
    elif "Static" == annotation_filename:
        class_id = 123
    elif "Unlabeled" == annotation_filename:
        class_id = 124
    return class_id


def each_sub_proc(file_name, dir_name, dataset_root, image_id, labels, each_image_json):
    file_name = file_name[:-4]
    instance_path = "{}/{}/instances/{}.png".format(
        dataset_root, dir_name, file_name)
    instance_image = Image.open(instance_path)
    instance_array = np.array(instance_image, dtype=np.uint16)

    image_label_instance_infomatrix = split_to_coco_creator(
        instance_array, labels)

    image_info = pycococreatortools.create_image_info(
        image_id, file_name + ".jpg", instance_image.size
    )
    each_image_json["images"].append(image_info)

    segmentation_id = 1
    for item in image_label_instance_infomatrix:
        class_id = convert_class_id(item["label_name"])
        category_info = {"id": class_id, "is_crowd": 1}
        binary_mask = item["image"]

        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id,
            image_id,
            category_info,
            binary_mask,
            instance_image.size,
            tolerance=2,
        )
        if annotation_info is not None:
            each_image_json["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1

    if each_image_json["images"] is []:
        print("Image {} doesn't contain one image".format(image_id))
    save_path = "{}/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, image_id
    )
    print("Saving to {}".format(save_path))
    with open(save_path, "w") as fp:
        json.dump(each_image_json, fp)


def load_datasets_and_proc(dataset_root, dir_name, files):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    each_image_json = dict({"images": [], "annotations": []})

    if(FULL_CPU):
        pool = mp.Pool(os.cpu_count())
    else:
        pool = mp.Pool(os.cpu_count()-4)

    with open("./config_Full.json") as config_file:
        config = json.load(config_file)

    labels = config["labels"]
    for idx, image_filename in enumerate(files):
        pool.apply_async(
            each_sub_proc,
            args=(
                image_filename,
                dir_name,
                dataset_root,
                idx + 1,
                labels,
                each_image_json,
            ),
        )

    pool.close()
    pool.join()


def readout_each_image(dataset_root, dir_name, seq):
    json_saved_path = "{}/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, seq
    )
    with open(json_saved_path) as fp:
        json_div = json.load(fp)

    return json_div


def main(dir_name, dataset_root, sample_type):
    dir_path = "{}/{}/instances".format(dataset_root, dir_name)
    files_list = []
    files = []
    i = 0
    for f in os.listdir(dir_path):
        if i > 100 and i % 600 == 1:
            files_list.append(files)
            files = []
        if f.endswith("png"):
            files.append(f)
            i += 1
            print("Loading image {}: {}".format(i, f))

    # Pre-create needed image paths
    if (
        os.path.exists(
            "{}/{}/massive_annotations".format(dataset_root, dir_name))
        is False
    ):
        os.makedirs("{}/{}/massive_annotations".format(dataset_root, dir_name))

    batch = 0
    if not files_list:
        load_datasets_and_proc(dataset_root, dir_name, files)
    else:
        for files in files_list:
            batch+=1
            print("Batch n°{}".format(batch))
            if batch > 0:
                load_datasets_and_proc(dataset_root, dir_name, files)

    combined_annotations = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    for idx in range(int(len(files))):
        each_image_json = readout_each_image(dataset_root, dir_name, idx + 1)
        combined_annotations["images"].extend(each_image_json["images"])
        combined_annotations["annotations"].extend(
            each_image_json["annotations"])

    combined_annotations["annotations"] = sorted(
        combined_annotations["annotations"],
        key=lambda item: item.__getitem__("image_id"),
    )

    for idx in range(len(combined_annotations["annotations"])):
        combined_annotations["annotations"][idx]["id"] = idx + 1

    combined_json_path = "{}/{}/instances_shape_{}2020.json".format(
        dataset_root, dir_name, sample_type
    )
    with open(combined_json_path, "w") as fp:
        json.dump(combined_annotations, fp)


if __name__ == "__main__":
    dataset_root = ""
    dir_name_train = "MapillaryVistas/training/v2.0"
    dir_name_val = "MapillaryVistas/validation/v2.0"
    main(dir_name_train, dataset_root, "training")
    main(dir_name_val, dataset_root, "validation")
