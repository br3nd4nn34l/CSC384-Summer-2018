# NOTE: this file contains functionality
# to run the auto-grader as a subprocess,
# so as to best emulate running it from the
# command line as per the instructions on the handout

# This is much easier to "play" from an IDE

import sys
import os
import subprocess

#region Generic Functions

def run_command(cmd_str, cwd, verbose=True):
    if verbose:
        print(cmd_str)

    subprocess.call(cmd_str, shell=True, cwd=cwd)

#endregion

# region Generic Functions for Running Python Scripts

def python_script_command(script_name, arg_string):
    return "\"{py}\" \"{script}\" {args}".format(
        py=sys.executable,
        script=script_name,
        args=arg_string
    )


def run_python_script(script_name, arg_string, cwd):
    cmd = python_script_command(script_name, arg_string)
    run_command(cmd, cwd)


# endregion

# region Functions for Running Pacman

def a1_code_path():
    a1_scripts_path = os.path.dirname(os.path.realpath(__file__))
    a1_code_rel_path = os.path.join(a1_scripts_path, "../code")
    return os.path.realpath(a1_code_rel_path)

def pacman_path():
    return os.path.join(a1_code_path(), "pacman.py")

def pacman_kwargs_str(kwargs_dict):
    if (not kwargs_dict):
        return ""

    sections = []
    for key in kwargs_dict:
        sections += ["{}={}".format(key, kwargs_dict[key])]
    return "-a {}".format(','.join(sections))

def pacman_timeout_arg_str(timeout):
    if not timeout:
        return ""
    else:
        return "-c --timeout={}".format(timeout)

def pacman_arg_str(maze_name, agent_name, timeout, kwargs_dict):
    return "-l {layout} -p {agent} {kwargs} {timeout}".format(
        layout=maze_name,
        agent=agent_name,
        kwargs=pacman_kwargs_str(kwargs_dict),
        timeout=pacman_timeout_arg_str(timeout),
    )

def run_pacman(maze_name, agent_name, timeout=False, kwargs_dict={}):
    run_python_script(
        pacman_path(),
        pacman_arg_str(maze_name, agent_name, timeout, kwargs_dict),
        a1_code_path()
    )

# endregion

#region Functions for Running Commands in Handout

def test_pacman_integrity():

    print("TESTING PACMAN INTEGRITY....")

    run_pacman("testMaze", "GoWestAgent")

    run_pacman("tinyMaze", "SearchAgent", kwargs_dict={
        "fn": "tinyMazeSearch"
    })

    print("PACMAN INTEGRITY TESTS COMPLETED.\n")

def test_q1():
    for maze in ["tinyMaze", "mediumMaze", "bigMaze"]:
        run_pacman(maze, "SearchAgent")

def test_eight_puzzle():
    run_python_script("eightpuzzle.py", "", a1_code_path())

def test_q2():
    test_eight_puzzle()

    for maze in ["tinyMaze", "mediumMaze", "bigMaze"]:
        run_pacman(maze, "SearchAgent", kwargs_dict={
            "fn" : "bfs"
        })


#endregion

if __name__ == '__main__':
    test_pacman_integrity()
    test_q2()

