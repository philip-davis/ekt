#ifndef _EKT_TELL_HASH_H
#define _EKT_TELL_HASH_H

#include<stdlib.h>
#include<string.h>

int hash_key(void *buf, size_t len, int mult, int mod)
{
    int sum = 0;
    int i;

    for(i = 0; i < len; i++) {
        sum += (i + 1) * ((char *)buf)[i];
    }

    return((sum * mult) % mod);
}

struct tell_hash_entry {
    void *buf;
    size_t len;
    int val;
    int idx1, idx2;
    struct tell_hash_entry *next;
};

struct tell_hash {
    struct tell_hash_entry **bins;
    size_t nbins;
    int salt;
};

struct tell_hash *create_thash(int size, int salt)
{
    struct tell_hash *th = malloc(sizeof(*th));
    int i;

    th->bins = calloc(sizeof(*th->bins), size);
    th->nbins = size;
    th->salt = salt;

   return(th); 
}

struct tell_hash_entry **find_thash_entry_p(struct tell_hash *th, void *buf, size_t len, int idx1, int idx2)
{
    int key = hash_key(buf, len, th->salt, th->nbins);
    struct tell_hash_entry **entry_p = &th->bins[key];
    struct tell_hash_entry *entry;
       
    while(*entry_p) {
        entry = *entry_p;
        if(idx1 == entry->idx1 && idx2 == entry->idx2 && len == entry->len && memcmp(buf, entry->buf, len) == 0) {
            return(entry_p);
        }
        entry_p = &entry->next;
    }
    return(entry_p);
}

int incr_thash_entry(struct tell_hash *th, void *buf, size_t len, int idx1, int idx2)
{
    struct tell_hash_entry **entry_p = find_thash_entry_p(th, buf, len, idx1, idx2);
    struct tell_hash_entry *entry = *entry_p;

    if(entry) {
        entry = *entry_p;
        entry->val++;
    } else {
        entry = malloc(sizeof(*entry));
        entry->buf = malloc(len);
        memcpy(entry->buf, buf, len);
        entry->val = 1;
        entry->len = len;
        entry->next = NULL;
        entry->idx1 = idx1;
        entry->idx2 = idx2;
        *entry_p = entry;
    }

    return(entry->val);
} 

int delete_thash_entry(struct tell_hash *th, void *buf, size_t len, int idx1, int idx2)
{
    struct tell_hash_entry **entry_p = find_thash_entry_p(th, buf, len, idx1, idx2); 
    struct tell_hash_entry *entry = *entry_p;

    if(!entry) {
        return(-1);
    }

    entry_p = &entry->next;
    free(entry->buf);
    free(entry);

    return(0);
}

#endif
