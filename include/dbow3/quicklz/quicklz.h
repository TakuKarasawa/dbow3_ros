#ifndef QUICKLZ_H_
#define QUICKLZ_H_

#include <string.h>

#ifndef QLZ_COMPRESSION_LEVEL
    #define QLZ_COMPRESSION_LEVEL 1
    //#define QLZ_COMPRESSION_LEVEL 2
    //#define QLZ_COMPRESSION_LEVEL 3

    #define QLZ_STREAMING_BUFFER 0
    //#define QLZ_STREAMING_BUFFER 100000
    //#define QLZ_STREAMING_BUFFER 1000000

    //#define QLZ_MEMORY_SAFE
#endif

#define QLZ_VERSION_MAJOR 1
#define QLZ_VERSION_MINOR 5
#define QLZ_VERSION_REVISION 0

#if QLZ_COMPRESSION_LEVEL != 1 && QLZ_COMPRESSION_LEVEL != 2 && QLZ_COMPRESSION_LEVEL != 3
    #error QLZ_COMPRESSION_LEVEL must be 1, 2 or 3
#endif

#if QLZ_COMPRESSION_LEVEL == 1
    #define QLZ_POINTERS 1
    #define QLZ_HASH_VALUES 4096
#elif QLZ_COMPRESSION_LEVEL == 2
    #define QLZ_POINTERS 4
    #define QLZ_HASH_VALUES 2048
#elif QLZ_COMPRESSION_LEVEL == 3
    #define QLZ_POINTERS 16
    #define QLZ_HASH_VALUES 4096
#endif

#if defined _LP64 || defined __LP64__ || defined __64BIT__ || _ADDR64 || defined _WIN64 || defined __arch64__ || __WORDSIZE == 64 || (defined __sparc && defined __sparcv9) || defined __x86_64 || defined __amd64 || defined __x86_64__ || defined _M_X64 || defined _M_IA64 || defined __ia64 || defined __IA64__
    #define QLZ_PTR_64
#endif

class QlzHashCompress
{
public:
#if QLZ_COMPRESSION_LEVEL == 1
    unsigned int cache;
#if defined QLZ_PTR_64 && QLZ_STREAMING_BUFFER == 0
    unsigned int offset;
#else
    const unsigned char *offset;
#endif
#else
    const unsigned char *offset[QLZ_POINTERS];
#endif

private:
};

class QlzHashDecompress
{
public:
#if QLZ_COMPRESSION_LEVEL == 1
    const unsigned char *offset;
#else
    const unsigned char *offset[QLZ_POINTERS];
#endif

private:
};

// states
class QlzStateCompress
{
public:
#if QLZ_STREAMING_BUFFER > 0
    unsigned char stream_buffer[QLZ_STREAMING_BUFFER];
#endif
    size_t stream_counter;
    QlzHashCompress hash[QLZ_HASH_VALUES];
    unsigned char hash_counter[QLZ_HASH_VALUES];

private:
};


#if QLZ_COMPRESSION_LEVEL == 1 || QLZ_COMPRESSION_LEVEL == 2
class QlzStateDecompress
{
public:
#if QLZ_STREAMING_BUFFER > 0
    unsigned char stream_buffer[QLZ_STREAMING_BUFFER];
#endif
    QlzHashDecompress hash[QLZ_HASH_VALUES];
    unsigned char hash_counter[QLZ_HASH_VALUES];
    size_t stream_counter;

private:
};
#elif QLZ_COMPRESSION_LEVEL == 3
class QlzStateDecompress
{
public:
#if QLZ_STREAMING_BUFFER > 0
    unsigned char stream_buffer[QLZ_STREAMING_BUFFER];
#endif
#if QLZ_COMPRESSION_LEVEL <= 2
    QlzHashDecompress hash[QLZ_HASH_VALUES];
#endif
    size_t stream_counter;
private:
};
#endif


#if defined (__cplusplus)
extern "C" {
#endif

// Public functions of QuickLZ
size_t qlz_size_decompressed(const char* source);
size_t qlz_size_compressed(const char* source);
size_t qlz_compress(const void* source,char* destination,size_t size,QlzStateCompress* state);
size_t qlz_decompress(const char* source,void* destination,QlzStateDecompress* state);
int qlz_get_setting(int setting);

#if defined (__cplusplus)
}
#endif

#endif  // QUICKLZ_H_
