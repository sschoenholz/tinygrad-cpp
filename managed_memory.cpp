#include "managed_memory.h"

#include <stdio.h>
#include <stdlib.h>

MemoryPool create_pool(u32 size, u32 chunk_size) {
	MemoryPool memory;

	if (chunk_size == 0)
		chunk_size = size;
	
	memory.size = size;
	memory.chunk_size = chunk_size;
	memory.chunk_count = (u32)ceil(size / ((f32)chunk_size));

	memory.current_chunk = 0;
	memory.current_in_chunk = 0;

	memory.data = (u8 **)malloc(sizeof(u8*) * memory.chunk_count);
	memset(memory.data, 0, sizeof(u8**) * memory.chunk_count);

	memory.data[0] = (u8*)malloc(sizeof(u8) * chunk_size);
	memset(memory.data[0], 0, sizeof(u8) * chunk_size);

	return memory;
}

MemoryPool create_pool(u8 *data, u32 size) {
	MemoryPool memory;

	memory.size = size;
	memory.chunk_size = size;
	memory.chunk_count = 1;

	memory.current_chunk = 0;
	memory.current_in_chunk = 0;

	memory.data = (u8 **)malloc(sizeof(u8*));
	memory.data[0] = data;

	return memory;
}

void clear(MemoryPool *M) {
	for (u32 i = 1; i < M->current_chunk && M->data[i]; i++) {
		free(M->data[i]);
		M->data[i] = NULL;
	}
	M->current_chunk = 0;
	M->current_in_chunk = 0;
}

void free(MemoryPool *M) {
	for (u32 i = 0; i < M->chunk_count && M->data[i] != NULL; i++)
		free(M->data[i]);

	free(M->data);

	M->data = NULL;
}

u8 *allocate(MemoryPool *M, u32 size) {
#ifdef PRINT_ALLOCATIONS
	std::cout << "Allocated " << size << " bytes. "
						<< memory->size - memory->current 
						<< " bytes remaining before allocation.\n";
#endif

	if (M->current_in_chunk + size > M->chunk_size)
	{
		M->current_chunk++;
		assert(M->current_chunk < M->chunk_count);

		M->data[M->current_chunk] = (u8*)malloc(sizeof(u8) * M->chunk_size);
	  memset(M->data[M->current_chunk], 0, sizeof(u8) * M->chunk_size);
		M->current_in_chunk = 0;
	}

	u8 *location = M->data[M->current_chunk] + M->current_in_chunk;
	memset(location, 0, size * sizeof(u8));
	M->current_in_chunk += size;

	return location;
}

u8* allocate_aligned(MemoryPool* M, u32 size, u32 alignment) {
#ifdef PRINT_ALLOCATIONS
	std::cout << "Allocated " << size << " bytes aligned to " << alignment << " bytes. "
		<< M->size - (M->current_chunk * M->chunk_size + M->current_in_chunk)
		<< " bytes remaining before allocation.\n";
#endif

	u8* unaligned_ptr = allocate(M, size + alignment - 1);

	if (((uintptr_t)unaligned_ptr & (alignment - 1)) == 0)
		return unaligned_ptr;

	u8* aligned_ptr = (u8*)(((uintptr_t)unaligned_ptr + alignment - 1) & ~(uintptr_t)(alignment - 1));
	// ptrdiff_t diff = aligned_ptr - unaligned_ptr;
  
	/*
	if (diff > 0) {
		// Shift the allocation pointer to the next aligned address in the current chunk
		M->current_in_chunk += diff;
	}
	*/

	return aligned_ptr;
}

u8 *allocate_no_clear(MemoryPool *M, u32 size) {
#ifdef PRINT_ALLOCATIONS
	std::cout << "Allocated " << size << " bytes. "
		<< memory->size - memory->current
		<< " bytes remaining before allocation.\n";
#endif

	if (M->current_in_chunk + size > M->chunk_size)
	{
		M->current_chunk++;
		assert(M->current_chunk < M->chunk_count);

		M->data[M->current_chunk] = (u8*)malloc(sizeof(u8) * M->chunk_size);
		memset(M->data[M->current_chunk], 0, sizeof(u8) * M->chunk_size);
		M->current_in_chunk = 0;
	}

	u8 *location = M->data[M->current_chunk] + M->current_in_chunk;
	M->current_in_chunk += size;

	return location;
}

MemoryPool g_temporary_memory;
MemoryPool g_level_memory;

u8 *t_allocate(u32 size) {
	return allocate(&g_temporary_memory, size);
}

u8 *t_allocate_no_clear(u32 size) {
	return allocate_no_clear(&g_temporary_memory, size);
}

MemoryChunkPool create_chunk_pool(u32 size, u32 chunk_size) {
	MemoryChunkPool memory;

	memory.size = size;
	memory.chunk_size = chunk_size;
	memory.chunk_count = (u32)ceil(size / ((f32)chunk_size));

	memory.current_chunk = 0;
	memory.current_in_chunk = 0;

	memory.data = (u8 **)malloc(sizeof(u8*) * memory.chunk_count);

	for (u32 i = 0; i < memory.chunk_count; i++) {
		memory.data[i] = (u8*)malloc(sizeof(u8) * chunk_size);
		memset(memory.data[i], 0, sizeof(u8) * chunk_size);
	}

	return memory;
}

void clear(MemoryChunkPool *M) {
  M->current_chunk = 0;
	M->current_in_chunk = 0;
}

void free(MemoryChunkPool *M) {
	for (u32 i = 0; i < M->chunk_count; i++)
		free(M->data[i]);

	free(M->data);

	M->data = NULL;
}

u8 *allocate(MemoryChunkPool *M, u32 size) {
#ifdef PRINT_ALLOCATIONS
	std::cout << "Allocated " << size << " bytes. "
		<< memory->size - memory->current << " bytes remaining before allocation.\n";
#endif

	if (M->current_in_chunk + size > M->chunk_size)
	{
		M->current_chunk++;
		M->current_in_chunk = 0;
	}

	assert(M->current_chunk < M->chunk_count);

	u8 *location = M->data[M->current_chunk] + M->current_in_chunk;
	memset(location, 0, size * sizeof(u8));
	M->current_in_chunk += size;

	return location;
}

u8 *allocate_no_clear(MemoryChunkPool *M, u32 size) {
#ifdef PRINT_ALLOCATIONS
	std::cout << "Allocated " << size << " bytes. "
		<< memory->size - memory->current << " bytes remaining before allocation.\n";
#endif

	if (M->current_in_chunk + size > M->chunk_size)
	{
		M->current_chunk++;
		M->current_in_chunk = 0;
	}

	assert(M->current_chunk < M->chunk_count);

	u8 *location = M->data[M->current_chunk] + M->current_in_chunk;
	M->current_in_chunk += size;

	return location;
}