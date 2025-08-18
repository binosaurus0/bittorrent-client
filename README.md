## Key Features

1. **Bencoding Support**: Complete decoder for parsing .torrent files
2. **Torrent File Parser**: Extracts metadata like announce URL, file info, and piece hashes
3. **Tracker Communication**: Async HTTP requests to get peer lists
4. **Peer Protocol**: Implements handshake, interested/unchoke, and piece requests
5. **Piece Management**: Handles pieces and blocks with integrity verification
6. **Async Architecture**: Uses asyncio for concurrent peer connections

## How to Use

1. **Install dependencies**:
```bash
pip install aiohttp
```

2. **Run the client**:
```bash
python bittorrent_client.py example.torrent
```

## Architecture Overview

- **BencodeDecoder**: Parses the binary .torrent file format
- **Torrent**: Wraps torrent metadata with convenient properties
- **Tracker**: Handles announce requests to get peer lists  
- **PeerConnection**: Manages TCP connections and protocol with individual peers
- **PieceManager**: Coordinates piece/block requests and file writing
- **BitTorrentClient**: Main coordinator that orchestrates everything

## Key Differences from Full Clients

This is a simplified educational implementation that:
- Only downloads (no seeding yet)
- Uses simple piece selection (sequential, not rarest-first)
- Limited to single-file torrents
- Basic error handling
