import Pipeline.Z_DataUnpacking as Z
import Pipeline.A_DataExtraction as A
import Pipeline.B_DataRepresentation as B
import Pipeline.C_LDA as C
import Pipeline.D_TextStructure as D
import Pipeline.E_AttentionModel as E
import Pipeline.F_BERT as F
import config as cf

def execute_pipeline(parts):
    if 'Z' in parts:
        Z.unpack_data()
    if 'A' in parts:
        pass
    if 'B' in parts:
        pass
    if 'C' in parts:
        pass
    if 'D' in parts:
        pass
    if 'E' in parts:
        pass
    if 'F' in parts:
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_pipeline(['Z'])
