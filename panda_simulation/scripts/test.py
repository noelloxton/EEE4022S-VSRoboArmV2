with open('moveit_command.py', 'r') as ms:
    for i, line in enumerate(ms):
        if '\xe2' in line:
            print(i, repr(line))
