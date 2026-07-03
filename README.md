# video_common

VideoCommon/
│
├── CMakeLists.txt
│
├── Common/
│   ├── Frame.h
│   ├── VideoTypes.h
│   ├── Protocol.h
│   └── DmaBuf.h
│
├── IPC/
│   ├── Uds.h
│   ├── Uds.cpp
│   ├── FdTransfer.h
│   └── FdTransfer.cpp
│
├── Client/
│   ├── VideoClient.h
│   └── VideoClient.cpp
│
└── Service/
    ├── VideoService.h
    ├── VideoService.cpp
    ├── Decoder.h
    └── Decoder.cpp