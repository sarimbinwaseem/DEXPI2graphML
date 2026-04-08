import os
from pathlib import Path

# import subprocess

import lxml
import functions
import pandas as pd
import PySimpleGUI as psg
from PIL import Image, ImageTk

# psg.theme('Default')

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_GRAPHML_COMPLETE_DIR = OUTPUT_DIR / "graphs_graphml" / "complete"
OUTPUT_GRAPHML_NO_PCE_DIR = OUTPUT_DIR / "graphs_graphml" / "noPCE"
OUTPUT_NODELIST_DIR = OUTPUT_DIR / "NodeLists"
OUTPUT_ERROR_DIR = OUTPUT_DIR / "errorLog"
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "graphs_plots"
LOGO_PATH = BASE_DIR / "GUI_figs" / "AD_Logo_EN_600dpi_gui.png"

if os.path.isfile(".folder_path"):
    with open(".folder_path", "r") as file:
        saved_dexpi_path = file.read()
else:
    saved_dexpi_path = ""


def list_plot_files() -> list[str]:
    return sorted(
        path.name for path in OUTPUT_PLOTS_DIR.iterdir() if path.suffix.lower() == ".png"
    )

# Define window content
col_left = [
    [psg.Text("Choose DEXPI - P&ID - folder...")],
    [
        psg.Input(key="path_dexpi", default_text=saved_dexpi_path),
        psg.FolderBrowse(initial_folder=saved_dexpi_path),
    ],
    [psg.Text("Processing Information / Console...")],
    [psg.Output(size=(60, 30), key="_output_")],
    [
        psg.Button("Convert"),
        psg.Button("show graphML P&ID in Explorer"),
        psg.Button("show Plot in Explorer"),
    ],
]

image_elem = psg.Image(
    size=(600, 450), key="plot_graph", visible=True, background_color="white"
)
list_elem = psg.Listbox(list_plot_files(), key="selected_plot", size=(50, 5))

col_right = [
    [psg.Text("Plot")],
    [image_elem],
    [list_elem],
    [
        psg.Button("P&ID-graph Plot"),
        psg.Button("show graphML"),
        psg.Button("show Error Log"),
    ],
]

col_bottom = [
    [psg.Image(str(LOGO_PATH))],
    [
        psg.Text(
            " CC: Technische Universität Dortmund, AG Apparatedesign \n Author: Jonas Oeing"
        )
    ],
]

layout = [
    [psg.Column(col_left), psg.Column(col_right, element_justification="c")],
    [psg.Column(col_bottom)],
]


# create window
window = psg.Window("dexpi2graph - ad@TUDO", layout)


def save_path(path: str) -> None:
    """Save inout folder path"""

    with open(".folder_path", "w") as file:
        file.write(path)


# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    list_elem.update(list_plot_files())
    if event == "Convert":

        dexpi_path = values["path_dexpi"]

        if dexpi_path == "":
            psg.popup("Enter path of the DEXPI folder!")

        else:
            dexpi_root = Path(dexpi_path)
            save_path(dexpi_path)
            print("Open Directory:", dexpi_root)
            print("Start Conversion of DEXPI files into GraphML...")

            for xml_path in sorted(dexpi_root.rglob("*.xml")):
                savename = "__".join(
                    xml_path.relative_to(dexpi_root).with_suffix("").parts
                ).replace(" ", "_")
                print(savename)
                functions.Dexpi2graph(
                    str(xml_path),
                    str(OUTPUT_GRAPHML_COMPLETE_DIR) + "/",
                    str(OUTPUT_GRAPHML_NO_PCE_DIR) + "/",
                    str(OUTPUT_NODELIST_DIR) + "/",
                    str(OUTPUT_ERROR_DIR) + "/",
                    savename,
                )
                functions.plot_graph2(
                    str(OUTPUT_GRAPHML_COMPLETE_DIR / (savename + ".xml")),
                    str(OUTPUT_PLOTS_DIR / savename),
                )
        list_elem.update(list_plot_files())

    ### Kasten auswahl einfügen
    if event == "show graphML P&ID in Explorer":
        # Application = os.getcwd()
        # Application = 'explorer "'+Application+'\Output\graphs_graphml\complete"'
        # subprocess.Popen(Application)

        psg.popup("Open Outputs/graphs_graphml/complete manually")

    if event == "P&ID-graph Plot":
        if values["selected_plot"] == []:
            psg.popup("Choose a P&ID-graph!")

        else:
            img = Image.open(OUTPUT_PLOTS_DIR / values["selected_plot"][0])
            img.thumbnail((600, 500))
            image_elem.update(data=ImageTk.PhotoImage(img), size=(600, 450))

    if event == "show Plot in Explorer":
        # Application = os.getcwd()
        # Application = 'explorer "'+Application+'\Output\graphs_plots"'
        # subprocess.Popen(Application)

        psg.popup("Open Output/graphs_plots manually")

    if event == "show graphML":
        if values["selected_plot"] == []:
            psg.popup("Choose a P&ID-graph!")

        else:
            window.FindElement("_output_").Update("")
            file = values["selected_plot"][0][:-4]
            xml_file = OUTPUT_GRAPHML_COMPLETE_DIR / (file + ".xml")
            tree = lxml.etree.parse(xml_file)
            pretty = lxml.etree.tostring(tree, encoding="unicode", pretty_print=True)
            print(pretty)

    if event == "show Error Log":
        if values["selected_plot"] == []:
            psg.popup("Choose a P&ID-graph!")

        else:
            window.FindElement("_output_").Update("")
            file = values["selected_plot"][0][:-4]
            error_file = OUTPUT_ERROR_DIR / (file + "_ErrorLog.xlsx")
            error_df = pd.read_excel(error_file)
            for i in range(0, len(error_df)):
                print(error_df["Warning"][i])
                print(error_df["Node(s)"][i])
                print("\n")

    # See if user wants to quit or window was closed
    if event == psg.WINDOW_CLOSED:
        break
    # Output a message to the window
