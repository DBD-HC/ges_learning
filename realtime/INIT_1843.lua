---------------------------------- STARTUP -------------------------------------
------------------------ DO NOT MODIFY THIS SECTION ----------------------------

-- mmwavestudio installation path
--RSTD_PATH = RSTD.GetRstdPath()

-- Declare the loading function
--dofile(RSTD_PATH .. "\\Scripts\\Startup.lua")

------------------------------ CONFIGURATIONS ----------------------------------
-- Use "DCA1000" for working with DCA1000
capture_device  = "DCA1000"

-- SOP mode
SOP_mode        = 2

-- RS232 connection baud rate
baudrate        = 115200
-- RS232 COM Port number
uart_com_port   = 4
-- Timeout in ms
timeout         = 1000

--system_ip = "192.168.33.30"
--fpga_ip = "192.168.33.180"

-- BSS firmware
bss_path        = "D:\\TI\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr68xx_radarss.bin"
-- MSS firmware
mss_path        = "D:\\TI\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr68xx_masterss.bin"

--adc_data_path   = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\test_data.bin"

-- Profile configuration
local profile_indx              =   0
local start_freq                =   60     -- GHz
local slope                     =   99.987     -- MHz/us
local idle_time                 =   340    -- us
local adc_start_time            =   5      -- us
local adc_samples               =   128-- Number of samples per chirp
local sample_freq               =   4000 -- ksps
local ramp_end_time             =   40     -- us
local rx_gain                   =   48     -- dB
local tx0OutPowerBackoffCode    =   0
local tx1OutPowerBackoffCode    =   0
local tx2OutPowerBackoffCode    =   0
local tx0PhaseShifter           =   0
local tx1PhaseShifter           =   0
local tx2PhaseShifter           =   0
local txStartTimeUSec           =   0
local hpfCornerFreq1            =   0      -- 0: 175KHz, 1: 235KHz, 2: 350KHz, 3: 700KHz
local hpfCornerFreq2            =   0      -- 0: 350KHz, 1: 700KHz, 2: 1.4MHz, 3: 2.8MHz

-- Frame configuration    
local start_chirp_tx            =   0
local end_chirp_tx              =   1
--local nchirp_loops              =   128     -- Number of chirps per frame
local nchirp_loops              =   64     -- Number of chirps per frame
--local nframes_master            =   100     -- Number of Frames for Master
local nframes_master            =   0  -- Number of Frames for Master
local nframes_slave             =   10     -- Number of Frames for Slaves
--local Inter_Frame_Interval      =   100    -- ms
local Inter_Frame_Interval      =   50    -- ms	
local trigger_delay             =   0      -- us  
local trig_list                 =   {1,2,2,2} -- 1: Software trigger, 2: Hardware trigger    


--ar1.ConfigureRFDCCard_EEPROM(system_ip, fpga_ip, "12:34:56:78:90:12", 4096, 4099)

------------------------- Connect Tab settings ---------------------------------
-- Select Capture device


-- SOP mode
ret=ar1.SOPControl(SOP_mode)
RSTD.Sleep(timeout)
if(ret~=0)
then
    print("******* SOP FAIL *******")
    return
end


-- RS232 Connect
ret=ar1.Connect(uart_com_port,baudrate,timeout)
RSTD.Sleep(timeout)
if(ret~=0)
then
    print("******* Connect FAIL *******")
    return
end
ar1.Calling_IsConnected()
ar1.SelectChipVersion("IWR6843")
-- ar1.SelectChipVersion("AR1642")
ar1.SelectChipVersion("IWR6843")
ar1.frequencyBandSelection("60G")


-- Download BSS Firmware
ret=ar1.DownloadBSSFw(bss_path)
RSTD.Sleep(2*timeout)
if(ret~=0)
then
    print("******* BSS Load FAIL *******")
    return
end

-- Download MSS Firmware
ret=ar1.DownloadMSSFw(mss_path)
RSTD.Sleep(2*timeout)
if(ret~=0)
then
    print("******* MSS Load FAIL *******")
    return
end


-- SPI Connect
--ar1.PowerOn(0, 1000, 0, 0)
ar1.PowerOn(0, 1000, 0, 0)

-- RF Power UP
ar1.RfEnable()

------------------------- Other Device Configuration ---------------------------

-- ADD Device Configuration here

ar1.ChanNAdcConfig(1, 0, 1, 1, 1, 1, 1, 2, 2, 0)

ar1.LPModConfig(0, 0)

ar1.RfInit()
RSTD.Sleep(1000)


ar1.DataPathConfig(513, 1216644097, 0)

--ar1.DataPathConfig(1, 1, 0)

ar1.LvdsClkConfig(1, 1)

ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)

--ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)

--ar1.SetTestSource(4, 3, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -2.5, 327, 327, 0, 0, 0, 0, -327, 0, -327, 
--                     327, 327, 327, -95, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0)
                  
--ar1.ProfileConfig(0, 77, 100, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 10000, 0, 0, 30)
ar1.ProfileConfig(0, start_freq, idle_time, adc_start_time, ramp_end_time, 0, 0, 0, 0, 0, 0, slope, 0, adc_samples, sample_freq, 0, 0, rx_gain)



-- Chirp 0
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)


-- Chirp 1
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 0, 1)



ar1.DisableTestSource(0)

--ar1.EnableTestSource(1)

ar1.FrameConfig(start_chirp_tx, end_chirp_tx, nframes_master, nchirp_loops, Inter_Frame_Interval, 0, 0, 1)
--ar1.FrameConfig(0, 0, 8, 128, 40, 0, 0, 1)


ret=ar1.SelectCaptureDevice(capture_device)
if(ret~=0)
then
    print("******* Wrong Capture device *******")
    return
end


--ar1.ConfigureRFDCCard_EEPROM("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4101)
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)

ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)

ar1.CaptureCardConfig_PacketDelay(5)

--Start Record ADC data
--ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
--RSTD.Sleep(1000)

--Trigger frame
--ar1.StartFrame()
--RSTD.Sleep(5000)


--Post process the Capture RAW ADC data
--ar1.StartMatlabPostProc(adc_data_path)
--WriteToLog("Please wait for a few seconds for matlab post processing .....!!!! \n", "green")
--RSTD.Sleep(10000)

------------------------- Close the Connection ---------------------------------
-- SPI disconnect
--ar1.PowerOff()

-- RS232 disconnect
--ar1.Disconnect()

------------------------- Exit MMwave Studio GUI -----------------------------------
--os.exit()

-- end
