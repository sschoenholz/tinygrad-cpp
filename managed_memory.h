#pragma once

#include "utilities.h"

struct MemoryPool {
  u32 size;
	u32 chunk_size;
	u32 chunk_count;

  u32 current_in_chunk;
	u32 current_chunk;

  bool b_locked;

  u8 **data;
};

MemoryPool create_pool(u32 size, u32 chunk_size=0);
MemoryPool create_pool(u8 *data, u32 size);

void clear(MemoryPool *memory);
void free(MemoryPool *memory);

u8 *allocate(MemoryPool *memory, u32 size);
u8* allocate_aligned(MemoryPool* M, u32 size, u32 alignment);
u8 *allocate_no_clear(MemoryPool *memory, u32 size);

template <class T>
T *allocate(MemoryPool *M, u32 count) {
	return (T*)allocate(M, sizeof(T) * count);
}

template <class T>
T *allocate(MemoryPool *M) {
	return (T*)allocate(M, sizeof(T));
}

template <class T>
T* allocate_aligned(MemoryPool* M, u32 count, u32 alignment) {
	return (T*)allocate_aligned(M, sizeof(T) * count, alignment);
}

template <class T>
T* allocate_aligned(MemoryPool* M, u32 alignment) {
	return (T*)allocate_aligned(M, sizeof(T), alignment);
}

template <class T>
T *allocate_no_clear(MemoryPool *M, u32 count) {
	return (T*)allocate_no_clear(M, sizeof(T) * count);
}

template <class T>
T *allocate_no_clear(MemoryPool *M) {
	return (T*)allocate_no_clear(M, sizeof(T));
}

extern MemoryPool g_temporary_memory;

u8 *t_allocate(u32 size);
u8 *t_allocate_no_clear(u32 size);

template <class T>
T* t_allocate() {
	return allocate<T>(&g_temporary_memory);
}

template <class T>
T *t_allocate(u32 count) {
	return allocate<T>(&g_temporary_memory, count);
}

template <class T>
T* t_allocate_aligned(u32 alignment) {
	return allocate_aligned<T>(&g_temporary_memory, alignment);
}

template <class T>
T* t_allocate_aligned(u32 count, u32 alignment) {
	return allocate_aligned<T>(&g_temporary_memory, count, alignment);
}

template <class T>
T *t_allocate_no_clear(u32 count) {
	return allocate_no_clear<T>(&g_temporary_memory, count);
}

struct MemoryChunkPool {
	u32 size;
	u32 chunk_size;
	u32 chunk_count;

	u32 current_in_chunk;
	u32 current_chunk;

	u8 **data;
};

MemoryChunkPool create_chunk_pool(u32 size, u32 chunk_size);

void clear(MemoryChunkPool *memory);
void free(MemoryChunkPool *memory);

u8 *allocate(MemoryChunkPool *memory, u32 size);
u8 *allocate_no_clear(MemoryChunkPool *memory, u32 size);

template <class T>
T *allocate(MemoryChunkPool *M, u32 count) {
	return (T*)allocate(M, sizeof(T) * count);
}

template <class T>
T *allocate(MemoryChunkPool *M) {
	return (T*)allocate(M, sizeof(T));
}

template <class T>
T *allocate_no_clear(MemoryChunkPool *M, u32 count) {
	return (T*)allocate_no_clear(M, sizeof(T) * count);
}

template <class T>
T *allocate_no_clear(MemoryChunkPool *M) {
	return (T*)allocate_no_clear(M, sizeof(T));
}
