# NOTE: this file contains functionality
# to run the auto-grader as a subprocess,
# so as to best emulate running it from the
# command line as per the instructions on the handout

# This is much easier to "play" from an IDE

import sys
import os
import subprocess
import re


# region Generic Functions

def run_command(cmd_str, cwd, verbose=True):
    if verbose:
        print(cmd_str)
    return subprocess.check_output(cmd_str, shell=True, cwd=cwd)


# endregion

# region Generic Functions for Running Python Scripts

def python_script_command(script_name, arg_string):
    return "\"{py}\" \"{script}\" {args}".format(
        py=sys.executable,
        script=script_name,
        args=arg_string
    )


def run_python_script(script_name, arg_string, cwd, verbose=True):
    cmd = python_script_command(script_name, arg_string)
    return run_command(cmd, cwd, verbose=verbose)


# endregion

# region Functions for Running Pacman

def a2_code_path():
    a2_scripts_path = os.path.dirname(os.path.realpath(__file__))
    a2_code_rel_path = os.path.join(a2_scripts_path, "../code")
    return os.path.realpath(a2_code_rel_path)


def pacman_path():
    return os.path.join(a2_code_path(), "pacman.py")


def autograder_path():
    return os.path.join(a2_code_path(), "autograder.py")


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


def pacman_num_ghosts_arg_str(num_ghosts):
    if num_ghosts:
        return "-k {}".format(num_ghosts)
    return ""


def pacman_no_graphics_arg_str(no_graphics):
    if no_graphics:
        return "-q"
    return ""


def pacman_num_games_arg_str(num_games):
    if num_games < 2:
        return ""
    return "-n {}".format(num_games)


def pacman_arg_str(maze_name, agent_name,
                   timeout, num_ghosts,
                   no_graphic, num_games,
                   kwargs_dict):
    return "-l {layout} -p {agent} {num_ghosts} {num_games} {kwargs} {timeout} {no_graphic}".format(
        layout=maze_name,
        agent=agent_name,
        num_games=pacman_num_games_arg_str(num_games),
        num_ghosts=pacman_num_ghosts_arg_str(num_ghosts),
        kwargs=pacman_kwargs_str(kwargs_dict),
        timeout=pacman_timeout_arg_str(timeout),
        no_graphic=pacman_no_graphics_arg_str(no_graphic)
    )


def run_pacman(maze_name, agent_name,
               no_graphic=False,
               timeout=False,
               num_ghosts=None,
               kwargs_dict={},
               num_games=1,
               verbose=True):
    return run_python_script(
        pacman_path(),
        pacman_arg_str(maze_name, agent_name, timeout, num_ghosts, no_graphic, num_games, kwargs_dict),
        a2_code_path(),
        verbose=verbose
    )


def run_autograder(question_name, graphics=True):
    graphics_str = "" if graphics else "--no-graphics"
    question_str = "-q {}".format(question_name)
    return run_python_script(
        autograder_path(),
        " ".join([question_str, graphics_str]),
        a2_code_path()
    )


# endregion

# region Functions for Running Commands in Handout

def test_q1():
    print(run_pacman(
        maze_name="testClassic",
        agent_name="ReflexAgent"
    ))

    print(run_pacman(
        maze_name="openClassic",
        agent_name="ReflexAgent"
    ))


def test_q2():
    print(run_pacman(
        maze_name="minimaxClassic",
        agent_name="MinimaxAgent",
        kwargs_dict={
            "depth": 4
        }
    ))

    # PacMan should try to commit soduku here
    print(run_pacman(
        maze_name="trappedClassic",
        agent_name="MinimaxAgent",
        kwargs_dict={
            "depth": 3
        }
    ))


def test_q3():
    print(run_pacman(
        agent_name="AlphaBetaAgent",
        maze_name="smallClassic",
        kwargs_dict={
            "depth": 3
        }
    ))


def test_q4():
    print(run_pacman(
        agent_name="ExpectimaxAgent",
        maze_name="minimaxClassic",
        kwargs_dict={
            "depth": 3
        }
    ))

    for agent in ["AlphaBetaAgent", "ExpectimaxAgent"]:
        print(run_pacman(
            agent_name=agent,
            maze_name="trappedClassic",
            kwargs_dict={
                "depth": 3
            },
            num_games=10
        ))


def test_q5():
    print(run_pacman(
        agent_name="ExpectimaxAgent",
        maze_name="smallClassic",
        num_games=10,
        kwargs_dict={
            "evalFn": "better"
        }
    ))


def evaluate_q1(num_games=100):
    result = run_pacman(
        maze_name="openClassic",
        agent_name="ReflexAgent",
        no_graphic=True,
        num_games=num_games
    )

    print(result)


def evaluate_q2(num_games=100):
    result = run_pacman(
        agent_name="MinimaxAgent",
        maze_name="minimaxClassic",
        no_graphic=True,
        num_games=num_games,
        kwargs_dict={
            "depth": 4
        }
    )

    print(result)


# endregion

if __name__ == '__main__':
    test_q5()
