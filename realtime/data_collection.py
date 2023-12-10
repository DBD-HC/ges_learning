"""
    Author: Yadong Li and Dong Zhang
    Email: yadongli@mail.ustc.edu.cn
    Function: Collecting AWR1843 adc data in real-time
"""
import os
import socket
import sys
import threading
import time
import numpy as np
import re
import _thread


frames_idx = []
def fetchNewFramesData(obj):
    newFramesData = np.array([])
    newFrameIndex = np.array([])
    flag = 0

    new_updated_frame_num = int(np.fix(obj['frameDataBuff_index'] / obj['oneFrameDataSize']));
    if new_updated_frame_num > 1:
        print('Udp Lose')

    if new_updated_frame_num > 0:
        flag = 1
        for i in range(0, new_updated_frame_num, 1):
            obj['frameId'] = obj['frameId'] + 1
            currFrameData = obj['frameDataBuff'][0:obj['oneFrameDataSize']]
            currFrameData = np.frombuffer(currFrameData.data, dtype=np.int16, count=int(currFrameData.size / 2),
                                          offset=0)
            obj['frameDataBuff'] = np.concatenate((obj['frameDataBuff'][obj['oneFrameDataSize']:], obj['zeroPadding']))
            obj['frameDataBuff_index'] = obj['frameDataBuff_index'] - obj['oneFrameDataSize']
            if i == 0:
                newFramesData = np.array([currFrameData])
            else :
                newFramesData = np.concatenate((newFramesData, currFrameData[None, :]))
            newFrameIndex = np.concatenate((newFrameIndex, np.array([obj['frameId']])))

    return flag, newFramesData, newFrameIndex


def readOneUdpPacket(obj, currUdpData):
    seqRatio = np.array([1, 256, 256 << 8, 256 << 16])
    dataSizeRatio = np.array([1, 256, 256 << 8, 256 << 16, 256 << 24, 256 << 32])
    flag = 0
    if currUdpData.size == 0:
        return flag

    currSeqNumArray = currUdpData[0:4]
    obj['currSeqNum'] = np.matmul(currSeqNumArray, seqRatio)

    dataSizeNumArray = currUdpData[4:10]
    obj['dataSizeTransed'] = np.matmul(dataSizeNumArray, dataSizeRatio)

    if obj['total_data_size'] != 0 and obj['dataSizeTransed'] == 0:
        return

    if obj['total_data_size'] != obj['dataSizeTransed']:
        lostDataSize = obj['dataSizeTransed'] - obj['total_data_size']
        obj['frameDataBuff'][obj['frameDataBuff_index']: obj['frameDataBuff_index'] + lostDataSize] = np.zeros(
            lostDataSize, dtype=np.uint8)
        obj['total_data_size'] = obj['dataSizeTransed']
        obj['frameDataBuff_index'] = obj['frameDataBuff_index'] + lostDataSize
        flag = 1

    currUdpRawData = currUdpData[10:]
    udp_packet_size = currUdpData.size
    raw_data_size = udp_packet_size - 10
    obj['total_data_size'] = obj['total_data_size'] + raw_data_size

    if obj['currSeqNum'] == 1:
        obj['prevUdpRawData'] = currUdpRawData
        return

    prev_raw_data_size = obj['prevUdpRawData'].size
    obj['frameDataBuff'][obj['frameDataBuff_index']: obj['frameDataBuff_index'] + prev_raw_data_size] = obj[
        'prevUdpRawData']
    obj['frameDataBuff_index'] = obj['frameDataBuff_index'] + prev_raw_data_size
    obj['prevUdpRawData'] = currUdpRawData


def getOneUdpPacket(filename, udp_index):
    currUdpData = 0
    udp_index += 1
    flag = 0

    fsize = os.path.getsize(filename)
    if udp_index > (fsize - 2000):
        prefix = np.array([], dtype=np.uint8)
        return flag, udp_index, currUdpData, prefix
    startCodeArray = np.fromfile(filename, dtype=np.uint8, count=4, offset=udp_index)
    if np.sum(np.equal(startCodeArray, [3, 2, 8, 0])) == 4:
        pl_Array = np.fromfile(filename, dtype=np.uint8, count=2, offset=udp_index + 4)
        packet_length = pl_Array[0] + pl_Array[1] * 256
        prefix = np.concatenate((np.array([3, 2, 8, 0], dtype=np.uint8), pl_Array))
        startCodeNextArray = np.fromfile(filename, dtype=np.uint8, count=4, offset=udp_index + packet_length + 6)
        if np.sum(np.equal(startCodeNextArray, [3, 2, 8, 0])) == 4:
            currUdpData = np.fromfile(filename, dtype=np.uint8, count=packet_length, offset=udp_index + 6)
            udp_index += packet_length + 5
            flag = 1
        else:
            flag = -1
    else:
        flag = -1

    return flag, udp_index, currUdpData, prefix



def radar_process_frame(radarObj, timeDomainData):
    range_fftsize = radarObj['range_fftsize']
    doppler_fftsize = radarObj['doppler_fftsize']
    angle_fftsize = radarObj['angle_fftsize']
    frameComplex = radarObj['frameComplex']
    frameComplexFinal = radarObj['frameComplexFinal']
    rangeFFTOut = radarObj['rangeFFTOut']
    DopplerFFTOut = radarObj['DopplerFFTOut']
    gAdcOneSampleSize = radarObj['gAdcOneSampleSize']
    numAdcSamples = radarObj['numAdcSamples']
    numRxChan = radarObj['numRxChan']
    numChirpsPerFrame = radarObj['numChirpsPerFrame']
    nLoopsIn1Frame = radarObj['nLoopsIn1Frame']
    nChirpsIn1Loop = radarObj['nChirpsIn1Loop']


    rawData4 = np.reshape(timeDomainData, (4, int(timeDomainData.size / 4)), order='F')
    rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
    rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
    frameCplx = rawDataI + 1j * rawDataQ
    frameCplxTemp = np.reshape(frameCplx, (numAdcSamples * numRxChan, numChirpsPerFrame), order='F')
    frameCplxTemp = np.transpose(frameCplxTemp, (1, 0))
    for jj in range(0, numChirpsPerFrame, 1):
        frameComplex[jj, :, :] = np.transpose(
            np.reshape(frameCplxTemp[jj, :], (numAdcSamples, numRxChan), order='F'), (1, 0))
    for nLoop in range(0, nLoopsIn1Frame, 1):
        for nChirp in range(0, nChirpsIn1Loop, 1):
            frameComplexFinal[nLoop, nChirp, :, :] = frameComplex[nLoop * nChirpsIn1Loop + nChirp, :, :]
    frameComplexFinalTmp = np.transpose(frameComplexFinal, (3, 0, 2, 1))

    return frameComplexFinalTmp


def process_data(filename, data_length, radar_info ):
    raw_frames_data = np.array([])
    first_packet_flag = 1
    udp_index = -1
    frame_num = 0
    udpPacketObj = {}
    udpPacketObj['total_data_size'] = 0
    udpPacketObj['frameDataBuff_index'] = 0
    udpPacketObj['frameDataBuff'] = np.empty(1000000, dtype=np.uint8)
    udpPacketObj['prevUdpRawData'] = 0

    radarObj = radar_info['radarObj']
    radarObj['gAdcOneSampleSize'] = 4
    radarObj['numAdcSamples'] = 128
    radarObj['numRxChan'] = 4
    radarObj['numChirpsPerFrame'] = 128
    radarObj['nLoopsIn1Frame'] = 64
    radarObj['nChirpsIn1Loop'] = 2
    # radarObj['range_fftsize'] = 128
    # radarObj['doppler_fftsize'] = 64
    # radarObj['angle_fftsize'] = 64
    radarObj['frameComplex'] = np.zeros(
        (radarObj['numChirpsPerFrame'], radarObj['numRxChan'], radarObj['numAdcSamples']), dtype=complex)
    radarObj['frameComplexFinal'] = np.zeros(
        (radarObj['nLoopsIn1Frame'], radarObj['nChirpsIn1Loop'], radarObj['numRxChan'], radarObj['numAdcSamples']),
        dtype=complex)
    # radarObj['rangeFFTOut'] = np.zeros(
    #     (radarObj['range_fftsize'], radarObj['nLoopsIn1Frame'], radarObj['numRxChan'], radarObj['nChirpsIn1Loop']),
    #     dtype=complex)
    # radarObj['DopplerFFTOut'] = np.zeros(
    #     (radarObj['range_fftsize'], radarObj['doppler_fftsize'], radarObj['numRxChan'], radarObj['nChirpsIn1Loop']),
    #     dtype=complex)

    dataSizeOneChirp = radarObj['gAdcOneSampleSize'] * radarObj['numAdcSamples'] * radarObj['numRxChan']
    dataSizeOneFrame = dataSizeOneChirp * radarObj['numChirpsPerFrame']
    udpPacketObj['oneFrameDataSize'] = dataSizeOneFrame
    udpPacketObj['frameId'] = 0
    udpPacketObj['zeroPadding'] = np.zeros(dataSizeOneFrame, dtype=np.uint8, order='C')

    last_seq_num = 0
    seqRatio = np.array([1, 256, 256 << 8, 256 << 16], dtype=np.uint32)
    recv_bytes = 0

    ipadress = ("192.168.33.30", int(4011))
    RECVBUFFSIZE = 50 * 1000 * 1000
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(ipadress)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECVBUFFSIZE)
    # val = struct.pack("Q", 15050)
    # udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, val)
    print('Server端已准备就绪！等待数据传输')

    while True:
        try:
            currUdpPacketData, ipaddr = udp_socket.recvfrom(2048)

            if not currUdpPacketData:
                print('客户端程序已退出！服务端即将断开')
                time.sleep(1)
                break
            if (first_packet_flag == 1):
                first_packet_flag = 0

                print('RECV RDR Data ing !')
                _thread.start_new_thread(close_mmwave_control, ("Thread-2",))

            currUdpPacketData = np.frombuffer(currUdpPacketData, dtype=np.uint8)
        except KeyboardInterrupt:
            print('服务端准备退出！')
            time.sleep(1)
            sys.exit()
            udp_socket.close()

        currSeqNumArray = currUdpPacketData[0:4]
        currSeqNum = np.matmul(currSeqNumArray, seqRatio)
        recv_bytes += len(currUdpPacketData)

        if currSeqNum != (last_seq_num + 1):
            print("misorder: ", [last_seq_num, currSeqNum])

        last_seq_num = currSeqNum

        readOneUdpPacket(udpPacketObj, currUdpPacketData)
        flag, newFramesData, newFrameIndex = fetchNewFramesData(udpPacketObj)

        if flag == 1:
            frame_num = frame_num + 1
            n_frames = newFrameIndex.size
            for ii in range(0, n_frames):
                raw_frame_data = radar_process_frame(radarObj, newFramesData[ii])#################
                # io.savemat('D:/realtimedata/' +'frame_{:02d}.mat'.format(total_framenum), {'rai_dynamic_64':data })


                global save_frame_num
                save_frame_num +=1

                if save_frame_num == 1:
                    raw_frames_data = np.array([raw_frame_data])
                else:
                    raw_frames_data = np.concatenate((raw_frames_data, raw_frame_data[None, :]))

                if raw_frames_data.shape[0] == data_length:

                    np.save(filename,raw_frames_data)

                    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    localIP = "192.168.33.30"
                    localPort = 4096
                    clientSocket.bind((localIP, localPort))

                    clientSocket.sendto(b'\x5a\xa5\x06\x00\x00\x00\xaa\xee', ('192.168.33.180', 4096))
                    clientSocket.close()
                    #print("end!! total frame num :",frame_num)



def close_mmwave_control( threadName):
    x = os.popen("netstat -ano | findstr 4096").read()
    xx = re.findall(r'\d+', x)
    print(x)
    cmdstr = "taskkill -PID " + str(xx[5]) + " -F"
    x = os.popen(cmdstr).read()
    print(x)
    print("mmWavestudio spy control closed successfully!")



if __name__ == '__main__':
    radar_info = {}
    radar_info['radarObj'] = {}
    radar_info['data'] = np.array([])
    radar_info['frame_num'] = 0

    global save_frame_num
    save_frame_num = 0

    filename = 'D:/s'
    data_length = 400

    th1 = threading.Thread(target=process_data, args=[filename, data_length , radar_info])
    th1.start()
