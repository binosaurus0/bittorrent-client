#!/usr/bin/env python3
"""
A simple BitTorrent client implementation in Python 3 using asyncio.
Based on the BitTorrent protocol specification.
"""

import asyncio
import hashlib
import random
import struct
import time
from collections import OrderedDict
from urllib.parse import urlencode
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BencodeDecoder:
    """Decoder for bencoded data"""
    
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def decode(self):
        """Decode bencoded data"""
        return self._decode()
    
    def _decode(self):
        """Internal decode method"""
        if self.index >= len(self.data):
            raise ValueError("Unexpected end of data")
        
        ch = chr(self.data[self.index])
        
        if ch == 'i':
            return self._decode_int()
        elif ch == 'l':
            return self._decode_list()
        elif ch == 'd':
            return self._decode_dict()
        elif ch.isdigit():
            return self._decode_string()
        else:
            raise ValueError(f"Invalid bencoded data at index {self.index}")
    
    def _decode_int(self):
        """Decode integer"""
        self.index += 1  # Skip 'i'
        end = self.data.find(b'e', self.index)
        if end == -1:
            raise ValueError("Invalid integer encoding")
        
        num_str = self.data[self.index:end].decode('ascii')
        self.index = end + 1
        return int(num_str)
    
    def _decode_string(self):
        """Decode string"""
        colon = self.data.find(b':', self.index)
        if colon == -1:
            raise ValueError("Invalid string encoding")
        
        length = int(self.data[self.index:colon].decode('ascii'))
        self.index = colon + 1
        
        if self.index + length > len(self.data):
            raise ValueError("String length exceeds data")
        
        string_data = self.data[self.index:self.index + length]
        self.index += length
        return string_data
    
    def _decode_list(self):
        """Decode list"""
        self.index += 1  # Skip 'l'
        result = []
        
        while self.index < len(self.data) and chr(self.data[self.index]) != 'e':
            result.append(self._decode())
        
        if self.index >= len(self.data):
            raise ValueError("Unterminated list")
        
        self.index += 1  # Skip 'e'
        return result
    
    def _decode_dict(self):
        """Decode dictionary"""
        self.index += 1  # Skip 'd'
        result = OrderedDict()
        
        while self.index < len(self.data) and chr(self.data[self.index]) != 'e':
            key = self._decode()
            if not isinstance(key, bytes):
                raise ValueError("Dictionary key must be a string")
            value = self._decode()
            result[key] = value
        
        if self.index >= len(self.data):
            raise ValueError("Unterminated dictionary")
        
        self.index += 1  # Skip 'e'
        return result


class Torrent:
    """Torrent file parser and data container"""
    
    def __init__(self, torrent_path):
        with open(torrent_path, 'rb') as f:
            self.data = BencodeDecoder(f.read()).decode()
        
        self.info = self.data[b'info']
        self.info_hash = hashlib.sha1(self._encode_info()).digest()
        
    def _encode_info(self):
        """Re-encode the info dict to calculate hash"""
        # Simple bencode encoder for the info dict
        return self._bencode(self.info)
    
    def _bencode(self, obj):
        """Simple bencoding implementation"""
        if isinstance(obj, int):
            return f'i{obj}e'.encode()
        elif isinstance(obj, bytes):
            return f'{len(obj)}:'.encode() + obj
        elif isinstance(obj, str):
            obj_bytes = obj.encode()
            return f'{len(obj_bytes)}:'.encode() + obj_bytes
        elif isinstance(obj, list):
            result = b'l'
            for item in obj:
                result += self._bencode(item)
            result += b'e'
            return result
        elif isinstance(obj, dict):
            result = b'd'
            for key in sorted(obj.keys()):
                result += self._bencode(key)
                result += self._bencode(obj[key])
            result += b'e'
            return result
        else:
            raise ValueError(f"Cannot encode type {type(obj)}")
    
    @property
    def announce(self):
        return self.data[b'announce'].decode()
    
    @property
    def name(self):
        return self.info[b'name'].decode()
    
    @property
    def length(self):
        return self.info[b'length']
    
    @property
    def piece_length(self):
        return self.info[b'piece length']
    
    @property
    def pieces(self):
        """Return list of piece hashes"""
        pieces_data = self.info[b'pieces']
        pieces = []
        for i in range(0, len(pieces_data), 20):
            pieces.append(pieces_data[i:i+20])
        return pieces


class TrackerResponse:
    """Tracker response parser"""
    
    def __init__(self, data):
        self.data = data
    
    @property
    def interval(self):
        return self.data.get(b'interval', 1800)
    
    @property
    def peers(self):
        """Parse compact peer list"""
        peers_data = self.data.get(b'peers', b'')
        peers = []
        
        for i in range(0, len(peers_data), 6):
            if i + 6 <= len(peers_data):
                ip_bytes = peers_data[i:i+4]
                port_bytes = peers_data[i+4:i+6]
                
                ip = '.'.join(str(b) for b in ip_bytes)
                port = struct.unpack('>H', port_bytes)[0]
                peers.append((ip, port))
        
        return peers


class Tracker:
    """Tracker communication handler"""
    
    def __init__(self, torrent, peer_id):
        self.torrent = torrent
        self.peer_id = peer_id
    
    async def announce(self, uploaded=0, downloaded=0, left=None):
        """Make announce request to tracker"""
        if left is None:
            left = self.torrent.length
        
        params = {
            'info_hash': self.torrent.info_hash,
            'peer_id': self.peer_id.encode(),
            'uploaded': uploaded,
            'downloaded': downloaded,
            'left': left,
            'port': 6881,
            'compact': 1
        }
        
        url = self.torrent.announce + '?' + urlencode(params)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        decoded = BencodeDecoder(data).decode()
                        return TrackerResponse(decoded)
                    else:
                        logger.error(f"Tracker request failed: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Tracker request error: {e}")
            return None


class PeerMessage:
    """Base class for peer protocol messages"""
    pass


class Handshake(PeerMessage):
    """Handshake message"""
    
    def __init__(self, info_hash, peer_id):
        self.info_hash = info_hash
        self.peer_id = peer_id
    
    def encode(self):
        """Encode handshake message"""
        protocol = b'BitTorrent protocol'
        reserved = b'\x00' * 8
        
        return (struct.pack('B', len(protocol)) + protocol + reserved + 
                self.info_hash + self.peer_id.encode())
    
    @classmethod
    def decode(cls, data):
        """Decode handshake message"""
        if len(data) < 68:
            return None
        
        pstrlen = data[0]
        if len(data) < 49 + pstrlen:
            return None
        
        info_hash = data[28:48]
        peer_id = data[48:68].decode('utf-8', errors='ignore')
        
        return cls(info_hash, peer_id)


class BitField(PeerMessage):
    """BitField message"""
    
    def __init__(self, bitfield):
        self.bitfield = bitfield


class Interested(PeerMessage):
    """Interested message"""
    
    def encode(self):
        return struct.pack('>IB', 1, 2)


class Request(PeerMessage):
    """Request message"""
    
    def __init__(self, index, begin, length):
        self.index = index
        self.begin = begin
        self.length = length
    
    def encode(self):
        return struct.pack('>IBIII', 13, 6, self.index, self.begin, self.length)


class Piece(PeerMessage):
    """Piece message"""
    
    def __init__(self, index, begin, data):
        self.index = index
        self.begin = begin
        self.data = data


class PeerConnection:
    """Manages connection to a single peer"""
    
    def __init__(self, peer, torrent, peer_id, piece_manager):
        self.peer = peer
        self.torrent = torrent
        self.peer_id = peer_id
        self.piece_manager = piece_manager
        self.reader = None
        self.writer = None
        self.choked = True
    
    async def connect(self):
        """Connect to peer and start communication"""
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.peer[0], self.peer[1]),
                timeout=10
            )
            
            logger.info(f"Connected to peer {self.peer[0]}:{self.peer[1]}")
            
            # Send handshake
            handshake = Handshake(self.torrent.info_hash, self.peer_id)
            self.writer.write(handshake.encode())
            await self.writer.drain()
            
            # Receive handshake
            handshake_data = await self.reader.read(68)
            peer_handshake = Handshake.decode(handshake_data)
            
            if not peer_handshake or peer_handshake.info_hash != self.torrent.info_hash:
                logger.warning(f"Invalid handshake from {self.peer[0]}:{self.peer[1]}")
                return
            
            # Send interested message
            interested = Interested()
            self.writer.write(interested.encode())
            await self.writer.drain()
            
            # Start message loop
            await self._message_loop()
            
        except Exception as e:
            logger.error(f"Connection error with {self.peer[0]}:{self.peer[1]}: {e}")
        finally:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
    
    async def _message_loop(self):
        """Main message processing loop"""
        buffer = b''
        
        while True:
            try:
                data = await asyncio.wait_for(self.reader.read(4096), timeout=30)
                if not data:
                    break
                
                buffer += data
                
                while len(buffer) >= 4:
                    message_length = struct.unpack('>I', buffer[:4])[0]
                    
                    if message_length == 0:  # Keep-alive
                        buffer = buffer[4:]
                        continue
                    
                    if len(buffer) < 4 + message_length:
                        break  # Wait for more data
                    
                    message_data = buffer[4:4 + message_length]
                    buffer = buffer[4 + message_length:]
                    
                    await self._handle_message(message_data)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout with peer {self.peer[0]}:{self.peer[1]}")
                break
            except Exception as e:
                logger.error(f"Message loop error: {e}")
                break
    
    async def _handle_message(self, data):
        """Handle incoming peer message"""
        if len(data) == 0:
            return
        
        message_id = data[0]
        
        if message_id == 1:  # Unchoke
            self.choked = False
            logger.info(f"Unchoked by {self.peer[0]}:{self.peer[1]}")
            await self._request_piece()
        
        elif message_id == 5:  # BitField
            bitfield = data[1:]
            self.piece_manager.add_peer_bitfield(self.peer, bitfield)
        
        elif message_id == 7:  # Piece
            if len(data) >= 9:
                index, begin = struct.unpack('>II', data[1:9])
                piece_data = data[9:]
                piece_msg = Piece(index, begin, piece_data)
                
                if self.piece_manager.add_block(piece_msg):
                    await self._request_piece()
    
    async def _request_piece(self):
        """Request next piece from peer"""
        if self.choked:
            return
        
        request = self.piece_manager.next_request(self.peer)
        if request:
            self.writer.write(request.encode())
            await self.writer.drain()


class Block:
    """Represents a block within a piece"""
    
    def __init__(self, piece_index, offset, length):
        self.piece_index = piece_index
        self.offset = offset
        self.length = length
        self.data = None
        self.received = False


class PieceManager:
    """Manages pieces and blocks for the torrent"""
    
    BLOCK_SIZE = 2**14  # 16KB
    
    def __init__(self, torrent):
        self.torrent = torrent
        self.pieces = self._initialize_pieces()
        self.peer_bitfields = {}
        self.downloaded = 0
        self.output_file = None
    
    def _initialize_pieces(self):
        """Initialize all pieces and their blocks"""
        pieces = []
        total_length = self.torrent.length
        piece_length = self.torrent.piece_length
        
        for i in range(len(self.torrent.pieces)):
            # Calculate piece size
            if i == len(self.torrent.pieces) - 1:
                # Last piece might be smaller
                current_piece_length = total_length - (i * piece_length)
            else:
                current_piece_length = piece_length
            
            # Create blocks for this piece
            blocks = []
            for j in range(0, current_piece_length, self.BLOCK_SIZE):
                block_length = min(self.BLOCK_SIZE, current_piece_length - j)
                blocks.append(Block(i, j, block_length))
            
            pieces.append({
                'index': i,
                'length': current_piece_length,
                'blocks': blocks,
                'complete': False,
                'hash': self.torrent.pieces[i]
            })
        
        return pieces
    
    def add_peer_bitfield(self, peer, bitfield):
        """Add peer's bitfield information"""
        self.peer_bitfields[peer] = bitfield
    
    def next_request(self, peer):
        """Get next block request for peer"""
        # Simple strategy: request first available block
        for piece in self.pieces:
            if piece['complete']:
                continue
            
            # Check if peer has this piece
            if not self._peer_has_piece(peer, piece['index']):
                continue
            
            for block in piece['blocks']:
                if not block.received:
                    return Request(piece['index'], block.offset, block.length)
        
        return None
    
    def _peer_has_piece(self, peer, piece_index):
        """Check if peer has specific piece"""
        if peer not in self.peer_bitfields:
            return False
        
        bitfield = self.peer_bitfields[peer]
        byte_index = piece_index // 8
        bit_index = piece_index % 8
        
        if byte_index >= len(bitfield):
            return False
        
        return bool(bitfield[byte_index] & (1 << (7 - bit_index)))
    
    def add_block(self, piece_msg):
        """Add received block data"""
        piece = self.pieces[piece_msg.index]
        
        # Find the corresponding block
        for block in piece['blocks']:
            if block.offset == piece_msg.begin:
                block.data = piece_msg.data
                block.received = True
                break
        
        # Check if piece is complete
        if all(block.received for block in piece['blocks']):
            if self._verify_piece(piece):
                piece['complete'] = True
                self._write_piece(piece)
                logger.info(f"Completed piece {piece['index']}")
                return True
        
        return False
    
    def _verify_piece(self, piece):
        """Verify piece integrity using SHA1 hash"""
        piece_data = b''.join(block.data for block in piece['blocks'])
        piece_hash = hashlib.sha1(piece_data).digest()
        return piece_hash == piece['hash']
    
    def _write_piece(self, piece):
        """Write completed piece to file"""
        if not self.output_file:
            self.output_file = open(self.torrent.name, 'wb')
        
        piece_data = b''.join(block.data for block in piece['blocks'])
        self.output_file.seek(piece['index'] * self.torrent.piece_length)
        self.output_file.write(piece_data)
        self.output_file.flush()
        
        self.downloaded += len(piece_data)
    
    @property
    def complete(self):
        """Check if download is complete"""
        return all(piece['complete'] for piece in self.pieces)
    
    def __del__(self):
        if self.output_file:
            self.output_file.close()


class BitTorrentClient:
    """Main BitTorrent client"""
    
    def __init__(self, torrent_path):
        self.torrent = Torrent(torrent_path)
        self.peer_id = self._generate_peer_id()
        self.tracker = Tracker(self.torrent, self.peer_id)
        self.piece_manager = PieceManager(self.torrent)
        self.peers = []
        self.connections = []
    
    def _generate_peer_id(self):
        """Generate unique peer ID"""
        return '-PC0001-' + ''.join([str(random.randint(0, 9)) for _ in range(12)])
    
    async def start(self):
        """Start the BitTorrent client"""
        logger.info(f"Starting download of {self.torrent.name}")
        logger.info(f"File size: {self.torrent.length} bytes")
        logger.info(f"Pieces: {len(self.torrent.pieces)}")
        
        try:
            while not self.piece_manager.complete:
                # Get peers from tracker
                response = await self.tracker.announce(
                    downloaded=self.piece_manager.downloaded,
                    left=self.torrent.length - self.piece_manager.downloaded
                )
                
                if response and response.peers:
                    logger.info(f"Got {len(response.peers)} peers from tracker")
                    
                    # Connect to peers (limit concurrent connections)
                    tasks = []
                    for peer in response.peers[:10]:  # Limit to 10 peers
                        if peer not in [conn.peer for conn in self.connections]:
                            conn = PeerConnection(peer, self.torrent, self.peer_id, self.piece_manager)
                            self.connections.append(conn)
                            tasks.append(asyncio.create_task(conn.connect()))
                    
                    if tasks:
                        # Wait for connections to establish or timeout
                        await asyncio.wait(tasks, timeout=30, return_when=asyncio.FIRST_COMPLETED)
                
                # Wait before next tracker request
                await asyncio.sleep(30)
            
            logger.info("Download completed!")
            
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
        except Exception as e:
            logger.error(f"Download error: {e}")


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python bittorrent_client.py <torrent_file>")
        return
    
    torrent_path = sys.argv[1]
    client = BitTorrentClient(torrent_path)
    await client.start()


if __name__ == '__main__':
    asyncio.run(main())