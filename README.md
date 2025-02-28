# AquaRenderer
This is a blender-project that loads NHFLOW simulations from .vtu files - such as the output from DiveMesh and REEF3D - into blender so that the simnulation can be rendered in the blenders render engines with blenders flexible artistic tools, that your simulation deserves  

# Pre-requirements:
- Blender
- Python [A version of python will be available with the blender instalation, without requireing you to download python]
- acces to terminal [Duh]

# Optional requiremenmts:
- VS Code

# HOW 2 - STEP BY STEP
1. Download repo: 
    * <> Code 
    - Option 1: git clone https://github.com/estilmk47/AquaRenderer.git 
    - Option 2: Downlaod zip

2. Setup the IDE and python environment:
    - Open a terminal in the main repo 
    - Create the environment:
        Make sure to use the same python version as your blender install
        - Use the terminal cmd: python -m venv <name_of_your_viritual_environment> (Recomended to use '.venv' or 'venv')
        - Use a similar terminal cmd: & "C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe" -m venv venv (This ensures packages you later install with pip is compatible with the python that is ran in blender)
    - Change the venv variable in the sccript.py to whatever you chose to name your environment (If you named it .venv you do not need to change this variable)
    - activate the viritual environment:
        - WINDOWS: <name_of_your_viritual_environment>/Script/activate
        - Other operating systems: "You know better then me, or can probably find it out faster then me"

3. Install python requirements:
    - NB: Ensure and resolve that the python environment is of the same version that your blender version is runnig [Probably ]   
    - When the venv is activated run:
        - pip install -r requirements.txt

4. Run the script from blender:
    - Open blender.
    - Navigate to the scripting window.
    - Delete the current scripts path as it is not linked to the script.py file on your computer
    - Open a new Script/Text 'script.py' [and rename it - good practice]
    - OPTIONAL:
        - Change the all the lines commented with TODO in the *if __name__ == '__main__':* part of the script if you do not want to run the example. (The very bottom of almost all python scripts that has it) 
        - Click on Window > Toggle System Console [This tab is very convenient for debugging, you may print directly to this terminal from your python scripts]
    - Run the script (Play button in scripting tab, *or*, Alt+P for hotkey-users)

Technically you only need to install the python packages after *sys.path.append(pyhton_env_path) [16]* as the other ones are either a part of python core or the python packages that came with Blender.
If you don't know which libraries these are [but you want to know], may I suggest looking in the path similar to C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\lib\site-packages on your computer.


# TECHNICAL NOTES:
- VS Code:
    - The bpy object is "quantom entageled" with the blender file. We download blender-stubs instead to make the IDE [VS Code] get acces to autocomplete which makes code dev much easier in these modern times. This means, on the other hand, that the script will not be able to run from your IDE as it does not have acces to the blender file [only the stbs]. Run the script from blender instead:
- BLENDER:
    - The base blender file [main.blend] is suppose to have a few assets "living inside the blender file" (eg geometry nodes and some objects). The script assumes these are pressent. If you modify the name of these the script may fail.
    - Blender only fetches the script you are running and the other dependant script that are not allready fetched. If you modify scripts that the main script depends on blender will not know unless you close and re-open the blender file. DO NOT waste time debugging code that is not even beeing run!
    - The Scripting window layout have been changed to be convenient for development with VS Code. After a change in VS Code a RED alert should be visible on the scripting tab. Click on it and 

- ENVIRONMENT:
    - When you create your viritual envoronment to download the extra site-packages ensure it is the of the same version as whatever your blender is running:
        - you can check the python version of blender by running the *python.exe* with the *--version flag* from the path similar to C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin