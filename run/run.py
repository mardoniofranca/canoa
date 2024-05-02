import subprocess
arg2 = "300" 
arg3 = "12"
 
# Run the called script with arguments
for i in range(0,37):
    arg1 = str(i*10)
    subprocess.run(['python', 'exec.py', arg1, arg2, arg3])