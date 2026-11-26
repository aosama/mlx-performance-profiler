# Vendored libraries

The report's interactive graph view runs on two JavaScript libraries. They
are copied here, unmodified, and compiled into `libmlx_perf.a` at build time
(`cmake/mlx_perf_embed_text_assets.cmake`). Every report therefore stays one
self-contained HTML file that opens offline, with nothing fetched from a
network.

| Library | Version | Licence | File | SHA-256 of the file |
|---|---|---|---|---|
| [Cytoscape.js](https://github.com/cytoscape/cytoscape.js) | 3.35.0 | MIT ([`cytoscape/LICENSE`](cytoscape/LICENSE)) | `cytoscape/cytoscape.min.js` (`dist/cytoscape.min.js` in the npm package) | `71fa9ebec21a7ca2c40cff6aacc30d1f6464e363662ad4b2c197e1adc770cd94` |
| [cytoscape-dagre](https://github.com/cytoscape/cytoscape.js-dagre) | 4.1.0 | MIT ([`cytoscape-dagre/LICENSE`](cytoscape-dagre/LICENSE)) | `cytoscape-dagre/cytoscape-dagre.min.js` (`dist/cytoscape-dagre.min.js`) | `001b3542c717f2084177aa58b73879986c2d2406017fa63968980677c056f279` |

cytoscape-dagre's bundle includes [dagre](https://github.com/dagrejs/dagre)
3.x, the layered graph layout, whose MIT licence is in
[`cytoscape-dagre/LICENSE.dagre`](cytoscape-dagre/LICENSE.dagre).

Both packages came from the npm registry; their tarballs matched the
registry's published checksums (`npm view <package>@<version> dist.shasum`).

## Updating a library

1. Download the new version into a scratch directory, never into this one:
   `npm pack cytoscape@<version>`. Check the tarball against
   `npm view cytoscape@<version> dist.shasum`.
2. Copy the same `dist/` file and the `LICENSE` over the ones here.
3. Update the table above: version and `shasum -a 256` of the copied file.
4. Rebuild and run the tests: `graph_view_section_test` refuses a library
   that contains `</script`, which would end the report's script element
   early. Then open a report and check the graph view in a browser.
