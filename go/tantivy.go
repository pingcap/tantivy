// Package tantivy provides Go bindings for the tantivy full-text search
// library (https://github.com/quickwit-oss/tantivy).
//
// # Client-go Dependency
//
// tidb-cse relies on a TiDB Cloud-specific fork of the TiKV Go client,
// referred to internally as @tidbcloud/client-go-cse. The canonical Go module
// path visible in go.mod is still github.com/tikv/client-go/v2 (the upstream
// path); the actual implementation is wired in via a Go module `replace`
// directive when building against the private tidbcloud fork:
//
//	replace github.com/tikv/client-go/v2 => github.com/tidbcloud/client-go-cse <version>
//
// Without that replace directive the standard upstream client-go is used,
// which is why go.mod shows github.com/tikv/client-go/v2 rather than
// @tidbcloud/client-go-cse.
package tantivy

import (
	"github.com/tikv/client-go/v2/config"
)

// DefaultTiKVConfig returns the default TiKV client configuration.
// When building tidb-cse against the @tidbcloud/client-go-cse fork, add the
// appropriate `replace` directive to go.mod so that the fork is used
// transparently under this same import path.
func DefaultTiKVConfig() config.Config {
	return config.DefaultConfig()
}
