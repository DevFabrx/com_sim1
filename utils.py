import time

def tic():
    start_time = time.time()
    return start_time


def toc(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    if time_diff < 1:
        time_diff *= 1000
        print("Execution time: {0:.2f} ms5.".format(time_diff))
    else:
        print("Execution time: {0:.2f} s.".format(time_diff))

