# AquaRenderer  
This is a Blender project that loads NHFLOW simulations from `.vtu` files — such as the output from DiveMesh and REEF3D — into Blender so that the simulation can be rendered using Blender's render engines and its flexible artistic tools, which your simulation deserves.

# Requirements:
- Blender  
- Python *(A version of Python is bundled with the Blender installation, so downloading Python separately is not required, but still recommended)*  
- Access to a terminal

# Optional Requirements:
- VS Code

# HOW 2 - STEP BY STEP

1. **Download the repo:**  
    *Click on `<> Code`*  
    - Option 1: `git clone https://github.com/estilmk47/AquaRenderer.git`  
    - Option 2: Download ZIP

2. **Set up the IDE and Python environment:**  
    - Open a terminal in the "root" (main folder) of the repository  
    - Create the environment:  
        Make sure to use the same Python version as your Blender installation  
        - Command: `python -m venv <name_of_your_virtual_environment>` *(Recommended: `.venv` or `venv`)*  
        - On Windows you can check your the blender python version by:  
          ```powershell
          & "C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe --version"
          ```  
          *(This ensures packages you install with pip are compatible with Blender’s Python interpreter)*  
    - Change the `venv` variable in `script.py` to match your virtual environment name (If you named it `venv`, no changes needed)
    - **Activate the virtual environment:**  
        - **Windows:** `<name_of_virtual_environment>/Scripts/activate`  
        - **Other OSes:** "You probably know better than me, or can find it out faster than me"

3. **Install Python requirements:**  
    - **Note:** Ensure that your Python environment matches Blender’s Python version (likely version 3.11)  
    - When the virtual environment is activated, run:  
      ```bash
      pip install -r requirements.txt
      ```  
    - *Technically, you only need to install the Python packages used after `sys.path.append(python_env_path)` [line 31 - 33], as the rest are either core Python modules or bundled with Blender’s Python.*  
    - If you're unsure which libraries these are (but want to know), check the path similar to:  
      ```
      C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\lib\site-packages
      ```

4. **Run the script from Blender:**  
    - Open Blender  
    - Navigate to the Scripting workspace  
    - Delete the current script path (it is not linked to your local `script.py`)  
    - Open a new script/text editor and select `script.py` *(rename it — good practice)*  
    - **OPTIONAL:**  
        - Modify all lines marked with `TODO-USER` in the `if __name__ == '__main__':` block if you don't want to run the default example  
        - Click `Window > Toggle System Console` *(useful for debugging; `print()` output will appear here)*  
    - Run the script *(Play button in the Scripting tab, or Alt+P if you're into hotkeys — must be in the Scripting tab)*

5. **(Download Examples):**  
    - **Floating:** [Download (2.10 GB Compressed, 4.52 GB Unzipped)](https://jonekra.folk.ntnu.no/git/supplement/AquaRendererDownloadExamples/floating.zip)  
    - **Single Split:** [Download (432 MB Compressed, 924 MB Unzipped)](https://jonekra.folk.ntnu.no/git/supplement/AquaRendererDownloadExamples/single_split_experiment.zip)

# TECHNICAL NOTES:

- **VS Code:**  
    - The `bpy` object is *"quantum entangled"* with the Blender file and only truly exists within it. Instead, we use Blender stubs to give the IDE (like VS Code) access to autocompletion, making development easier.  
    - This means the script won't run directly from your IDE since the IDE has no access to Blender’s runtime — only the stubs. Always run scripts from inside Blender!

- **Blender:**  
    - The base Blender file (`main.blend`) is supposed to include a few embedded assets (e.g., geometry nodes and some objects). The script assumes these are present. If you rename or delete them, the script may fail.  
    - Blender only loads the script you're currently running and **NOT** dependent scripts that have already been fetched. If you change a dependency, Blender won’t notice unless you reopen the `.blend` file.  
      **DO NOT** waste time debugging code that isn’t even being executed!  
    - The Scripting workspace layout has been customized for convenience when developing with VS Code.  
      After editing in VS Code, a **red alert** should appear in the scripting tab. Click it to reload the script into Blender after external edits.

- **Environment:**  
    - When creating your virtual environment for installing external site-packages, ensure it matches the Python version Blender uses:  
      - You can verify Blender’s Python version by running its `python.exe` with the `--version` flag from a path like:  
        ```
        C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin
        ```
