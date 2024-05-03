import subprocess
arg2 = "50"  #N iter
arg3 = "8"   #Size 
 
# Run the called script with arguments
for i in range(0,361):
    arg1 = str(i)
    subprocess.run(['python', 'exec.py', arg1, arg2, arg3])
    break;