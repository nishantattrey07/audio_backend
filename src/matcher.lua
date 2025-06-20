-- matcher.lua
--
-- A Redis Lua script for high-performance audio fingerprint matching.
-- This script performs the entire "scatter-gather-histogram" process on the server side.
--
-- @param KEYS[1] A unique key for a temporary Redis Hash to store the histogram scores.
-- @param ARGV A flattened list of the sample's hashes and their anchor times.
--             It is structured as {hash1, time1, hash2, time2, ...}
-- @return The total number of potential matches found across all hashes.

-- The key for our temporary scoresheet, e.g., "match_scores:some-uuid"
local histogram_key = KEYS[1]

-- A counter for the total number of raw matches found.
local matches_found = 0

-- Loop through the arguments, two at a time.
-- #ARGV is the total number of arguments. We step by 2.
for i = 1, #ARGV, 2 do
    local sample_hash_key = ARGV[i]
    local sample_anchor_time = tonumber(ARGV[i+1])

    -- For each sample hash, get the list of "track_id:offset" strings from the database.
    -- This is an internal, fast call, not a network call.
    local db_matches = redis.call('LRANGE', sample_hash_key, 0, -1)

    -- If matches were found for this hash, process them.
    if #db_matches > 0 then
        -- Loop through each database match.
        for _, db_match_string in ipairs(db_matches) do
            -- Lua's equivalent of Python's .split(':')
            local track_id_str, db_anchor_time_str = db_match_string:match("([^:]+):([^:]+)")

            if track_id_str and db_anchor_time_str then
                local track_id = tonumber(track_id_str)
                local db_anchor_time = tonumber(db_anchor_time_str)

                -- Calculate the time difference (delta).
                local delta_t = db_anchor_time - sample_anchor_time

                -- Create the field key for our histogram, e.g., "101:2101".
                local field_key = track_id .. ':' .. delta_t

                -- Atomically increment the score for this specific track and time alignment.
                -- This is the core of the server-side histogramming.
                redis.call('HINCRBY', histogram_key, field_key, 1)

                matches_found = matches_found + 1
            end
        end
    end
end

-- Set an expiry on the temporary key just in case the client crashes
-- and fails to delete it. 60 seconds is a safe value.
redis.call('EXPIRE', histogram_key, 60)

-- Return the total number of raw matches we processed.
return matches_found