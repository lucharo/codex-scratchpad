import { ChurnData } from './churnAnalyzer';

/**
 * Cache manager for churn analysis results.
 * Uses a 5-minute TTL with explicit invalidation on file save.
 */
export class CacheManager {
    private cache = new Map<string, ChurnData>();
    private maxAge = 5 * 60 * 1000;  // 5 minutes TTL

    /**
     * Get cached churn data for a file path.
     * Returns null if not cached or if cache entry is stale.
     */
    get(filePath: string): ChurnData | null {
        const data = this.cache.get(filePath);
        if (!data) {
            return null;
        }

        // Check if entry is stale
        const age = Date.now() - data.analyzedAt.getTime();
        if (age > this.maxAge) {
            this.cache.delete(filePath);
            return null;
        }

        return data;
    }

    /**
     * Store churn data in the cache
     */
    set(filePath: string, data: ChurnData): void {
        this.cache.set(filePath, data);
    }

    /**
     * Invalidate cache for a specific file.
     * Called when a file is saved (content changed).
     */
    invalidate(filePath: string): void {
        this.cache.delete(filePath);
    }

    /**
     * Clear all cached data.
     * Called on manual refresh or settings change.
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * Get the number of cached entries (for debugging)
     */
    get size(): number {
        return this.cache.size;
    }
}
