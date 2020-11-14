import Preprocess as preprocess
import LDA as lda
import Score as score
import Visualize as visualize

if __name__ == '__main__':
    start = 5
    end = 25
    increment = 1
    runs = [0, 0, 1, 0]
    if runs[0]:
        preprocess.main()
    if runs[1]:
        lda.main(start, end, increment)
    if runs[2]:
        score.main(start, end, increment)
    if runs[3]:
        visualize.main(start, end, increment)
