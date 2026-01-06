/**
 * Cache manager for churn analysis results.
 *
 * Implements TTL-based caching with explicit invalidation
 * to avoid unnecessary recomputation while staying fresh.
 */

import * as vscode from 'vscode';
import { ChurnData } from './types';

/** Default cache TTL: 5 minutes */
const DEFAULT_MAX_AGE_MS = 5 * 60 * 1000;

/** Maximum cache entries to prevent memory bloat */
const MAX_CACHE_ENTRIES = 50;

interface CacheEntry {
    data: ChurnData;
    accessedAt: number;
}

/**
 * LRU cache for churn analysis results with TTL support.
 * Implements Disposable for proper cleanup.
 */
export class CacheManager implements vscode.Disposable {
    private readonly cache = new Map<string, CacheEntry>();
    private readonly maxAge: number;
    private cleanupInterval: NodeJS.Timeout | null = null;

    constructor(maxAgeMs: number = DEFAULT_MAX_AGE_MS) {
        this.maxAge = maxAgeMs;

        // Periodic cleanup of stale entries
        this.cleanupInterval = setInterval(
            () => this.cleanup(),
            this.maxAge
        );
    }

    /**
     * Get cached churn data for a file path.
     * Returns null if not cached or if entry is stale.
     */
    get(filePath: string): ChurnData | null {
        const entry = this.cache.get(filePath);
        if (!entry) {
            return null;
        }

        // Check if entry is stale
        const age = Date.now() - entry.data.analyzedAt.getTime();
        if (age > this.maxAge) {
            this.cache.delete(filePath);
            return null;
        }

        // Update access time for LRU
        entry.accessedAt = Date.now();
        return entry.data;
    }

    /**
     * Store churn data in the cache.
     * Enforces max entries limit using LRU eviction.
     */
    set(filePath: string, data: ChurnData): void {
        // Evict oldest entry if at capacity
        if (this.cache.size >= MAX_CACHE_ENTRIES && !this.cache.has(filePath)) {
            this.evictOldest();
        }

        this.cache.set(filePath, {
            data,
            accessedAt: Date.now(),
        });
    }

    /**
     * Invalidate cache for a specific file.
     * Called when a file is saved (content changed).
     */
    invalidate(filePath: string): void {
        this.cache.delete(filePath);
    }

    /**
     * Invalidate all entries for files within a directory.
     * Useful when git operations affect multiple files.
     */
    invalidateDirectory(dirPath: string): void {
        const normalizedDir = dirPath.endsWith('/') ? dirPath : `${dirPath}/`;
        for (const key of this.cache.keys()) {
            if (key.startsWith(normalizedDir)) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Clear all cached data.
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * Get cache statistics for debugging/monitoring.
     */
    getStats(): { size: number; maxAge: number; maxEntries: number } {
        return {
            size: this.cache.size,
            maxAge: this.maxAge,
            maxEntries: MAX_CACHE_ENTRIES,
        };
    }

    /**
     * Check if a file has cached data (without retrieving).
     */
    has(filePath: string): boolean {
        const entry = this.cache.get(filePath);
        if (!entry) {
            return false;
        }

        const age = Date.now() - entry.data.analyzedAt.getTime();
        if (age > this.maxAge) {
            this.cache.delete(filePath);
            return false;
        }

        return true;
    }

    /**
     * Remove stale entries from cache.
     */
    private cleanup(): void {
        const now = Date.now();
        for (const [key, entry] of this.cache.entries()) {
            const age = now - entry.data.analyzedAt.getTime();
            if (age > this.maxAge) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Evict the least recently accessed entry.
     */
    private evictOldest(): void {
        let oldestKey: string | null = null;
        let oldestAccess = Infinity;

        for (const [key, entry] of this.cache.entries()) {
            if (entry.accessedAt < oldestAccess) {
                oldestAccess = entry.accessedAt;
                oldestKey = key;
            }
        }

        if (oldestKey !== null) {
            this.cache.delete(oldestKey);
        }
    }

    /**
     * Dispose of cache resources.
     */
    dispose(): void {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
        this.cache.clear();
    }
}
