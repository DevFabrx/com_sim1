import time
import logging
from matplotlib.backends.backend_pdf import PdfPages


def tic():
    start_time = time.time()
    return start_time


def toc(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    if time_diff < 1:
        time_diff *= 1000
        print("Total execution time: {0:.2f} ms.".format(time_diff))
    else:
        print("Total execution time: {0:.2f} s.".format(time_diff))

def save_fig_to_pdf(pdf_name, *args):
    with PdfPages(pdf_name) as pdf:
        for figure in args:
            pdf.savefig(figure)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger = logging.getLogger('main')
        logger.info("{} took {:.2f} ms".format(func.__name__, (end-start)*1000))
        return result
    return wrapper
