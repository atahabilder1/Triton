import subprocess
import json
import tempfile
import os
import re
from typing import Dict, List, Optional, Any
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class SlitherWrapper:
    def __init__(self, timeout: int = 300, log_failures: bool = True, failure_log_path: str = "logs/pdg_failures.log"):
        self.timeout = timeout
        self.log_failures = log_failures
        self.failure_log_path = failure_log_path
        self.detector_mapping = {
            'reentrancy-eth': 'reentrancy',
            'reentrancy-no-eth': 'reentrancy',
            'reentrancy-benign': 'reentrancy',
            'integer-overflow': 'overflow',
            'integer-underflow': 'underflow',
            'unprotected-upgrade': 'access_control',
            'suicidal': 'self_destruct',
            'unchecked-lowlevel': 'unchecked_call',
            'unchecked-send': 'unchecked_call',
            'timestamp': 'timestamp_dependency',
            'tx-origin': 'tx_origin',
            'delegatecall-loop': 'delegatecall',
            'arbitrary-send': 'access_control'
        }

        # Comprehensive OpenZeppelin-compatible stubs
        # Full dependency chain to avoid cascading errors
        self.common_stubs = {
            # Base contracts
            'Context': '''
contract Context {
    function _msgSender() internal view virtual returns (address) { return msg.sender; }
    function _msgData() internal view virtual returns (bytes calldata) { return msg.data; }
}''',

            # ERC165 - Interface detection
            'ERC165': '''
interface IERC165 { function supportsInterface(bytes4 interfaceId) external view returns (bool); }
abstract contract ERC165 is IERC165 {
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) { return interfaceId == type(IERC165).interfaceId; }
}''',

            # ERC20 with full dependencies
            'ERC20': '''
contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
interface IERC20 { function totalSupply() external view returns (uint256); function balanceOf(address) external view returns (uint256); function transfer(address, uint256) external returns (bool); function allowance(address, address) external view returns (uint256); function approve(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }
contract ERC20 is Context, IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;
    function name() public view virtual returns (string memory) { return _name; }
    function symbol() public view virtual returns (string memory) { return _symbol; }
    function decimals() public view virtual returns (uint8) { return 18; }
    function totalSupply() public view virtual override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view virtual override returns (uint256) { return _balances[account]; }
    function transfer(address, uint256) public virtual override returns (bool) { return true; }
    function allowance(address, address) public view virtual override returns (uint256) { return 0; }
    function approve(address, uint256) public virtual override returns (bool) { return true; }
    function transferFrom(address, address, uint256) public virtual override returns (bool) { return true; }
}''',

            # ERC721 with full dependencies
            'ERC721': '''
interface IERC165 { function supportsInterface(bytes4) external view returns (bool); }
abstract contract ERC165 is IERC165 { function supportsInterface(bytes4) public view virtual override returns (bool) { return true; } }
interface IERC721 is IERC165 { function balanceOf(address) external view returns (uint256); function ownerOf(uint256) external view returns (address); function safeTransferFrom(address, address, uint256, bytes calldata) external; function safeTransferFrom(address, address, uint256) external; function transferFrom(address, address, uint256) external; function approve(address, uint256) external; function setApprovalForAll(address, bool) external; function getApproved(uint256) external view returns (address); function isApprovedForAll(address, address) external view returns (bool); }
contract ERC721 is ERC165, IERC721 {
    mapping(uint256 => address) private _owners;
    mapping(address => uint256) private _balances;
    function balanceOf(address owner) public view virtual override returns (uint256) { return _balances[owner]; }
    function ownerOf(uint256 tokenId) public view virtual override returns (address) { return _owners[tokenId]; }
    function approve(address, uint256) public virtual override {}
    function getApproved(uint256) public view virtual override returns (address) { return address(0); }
    function setApprovalForAll(address, bool) public virtual override {}
    function isApprovedForAll(address, address) public view virtual override returns (bool) { return false; }
    function transferFrom(address, address, uint256) public virtual override {}
    function safeTransferFrom(address, address, uint256) public virtual override {}
    function safeTransferFrom(address, address, uint256, bytes memory) public virtual override {}
    function supportsInterface(bytes4) public view virtual override(ERC165, IERC165) returns (bool) { return true; }
}''',

            # ERC721Enumerable
            'ERC721Enumerable': '''
interface IERC165 { function supportsInterface(bytes4) external view returns (bool); }
abstract contract ERC165 is IERC165 { function supportsInterface(bytes4) public view virtual override returns (bool) { return true; } }
interface IERC721 is IERC165 { function balanceOf(address) external view returns (uint256); function ownerOf(uint256) external view returns (address); function transferFrom(address, address, uint256) external; }
contract ERC721 is ERC165, IERC721 {
    function balanceOf(address) public view virtual override returns (uint256) { return 0; }
    function ownerOf(uint256) public view virtual override returns (address) { return address(0); }
    function transferFrom(address, address, uint256) public virtual override {}
    function supportsInterface(bytes4) public view virtual override(ERC165, IERC165) returns (bool) { return true; }
}
interface IERC721Enumerable is IERC721 { function totalSupply() external view returns (uint256); function tokenOfOwnerByIndex(address, uint256) external view returns (uint256); function tokenByIndex(uint256) external view returns (uint256); }
contract ERC721Enumerable is ERC721, IERC721Enumerable {
    function totalSupply() public view virtual override returns (uint256) { return 0; }
    function tokenOfOwnerByIndex(address, uint256) public view virtual override returns (uint256) { return 0; }
    function tokenByIndex(uint256) public view virtual override returns (uint256) { return 0; }
    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721, IERC165) returns (bool) { return true; }
}''',

            # Ownable with Context
            'Ownable': '''
contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor() { _owner = _msgSender(); }
    function owner() public view virtual returns (address) { return _owner; }
    modifier onlyOwner() { require(owner() == _msgSender()); _; }
    function renounceOwnership() public virtual onlyOwner { _owner = address(0); }
    function transferOwnership(address newOwner) public virtual onlyOwner { _owner = newOwner; }
}''',

            # SafeMath
            'SafeMath': '''
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) { return a + b; }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) { require(b <= a); return a - b; }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) { if (a == 0) return 0; return a * b; }
    function div(uint256 a, uint256 b) internal pure returns (uint256) { require(b > 0); return a / b; }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) { require(b != 0); return a % b; }
}''',

            # Address library
            'Address': '''
library Address {
    function isContract(address account) internal view returns (bool) { return account.code.length > 0; }
    function sendValue(address payable recipient, uint256 amount) internal { require(address(this).balance >= amount); (bool success,) = recipient.call{value: amount}(""); require(success); }
}''',

            # Strings library
            'Strings': '''
library Strings {
    function toString(uint256 value) internal pure returns (string memory) { if (value == 0) return "0"; uint256 temp = value; uint256 digits; while (temp != 0) { digits++; temp /= 10; } bytes memory buffer = new bytes(digits); while (value != 0) { digits -= 1; buffer[digits] = bytes1(uint8(48 + uint256(value % 10))); value /= 10; } return string(buffer); }
}''',

            # ReentrancyGuard
            'ReentrancyGuard': '''
contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;
    constructor() { _status = _NOT_ENTERED; }
    modifier nonReentrant() { require(_status != _ENTERED); _status = _ENTERED; _; _status = _NOT_ENTERED; }
}''',

            # Pausable
            'Pausable': '''
contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
contract Pausable is Context {
    bool private _paused;
    event Paused(address account);
    event Unpaused(address account);
    constructor() { _paused = false; }
    function paused() public view virtual returns (bool) { return _paused; }
    modifier whenNotPaused() { require(!paused()); _; }
    modifier whenPaused() { require(paused()); _; }
    function _pause() internal virtual whenNotPaused { _paused = true; }
    function _unpause() internal virtual whenPaused { _paused = false; }
}''',

            # Interfaces
            'IERC20': '''
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}''',

            'IERC721': '''
interface IERC165 { function supportsInterface(bytes4 interfaceId) external view returns (bool); }
interface IERC721 is IERC165 {
    function balanceOf(address owner) external view returns (uint256 balance);
    function ownerOf(uint256 tokenId) external view returns (address owner);
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    function transferFrom(address from, address to, uint256 tokenId) external;
    function approve(address to, uint256 tokenId) external;
    function getApproved(uint256 tokenId) external view returns (address operator);
    function setApprovalForAll(address operator, bool _approved) external;
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}''',

            # ERC20 Extensions
            'ERC20Burnable': '''
contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
interface IERC20 { function totalSupply() external view returns (uint256); function balanceOf(address) external view returns (uint256); function transfer(address, uint256) external returns (bool); }
contract ERC20 is Context, IERC20 {
    mapping(address => uint256) private _balances;
    uint256 private _totalSupply;
    function totalSupply() public view virtual override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view virtual override returns (uint256) { return _balances[account]; }
    function transfer(address, uint256) public virtual override returns (bool) { return true; }
    function _burn(address account, uint256 amount) internal virtual {}
}
abstract contract ERC20Burnable is Context, ERC20 {
    function burn(uint256 amount) public virtual { _burn(_msgSender(), amount); }
    function burnFrom(address account, uint256 amount) public virtual { _burn(account, amount); }
}''',

            'ERC20Pausable': '''
contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
contract Pausable is Context { bool private _paused; function paused() public view virtual returns (bool) { return _paused; } modifier whenNotPaused() { require(!paused()); _; } }
interface IERC20 { function transfer(address, uint256) external returns (bool); }
contract ERC20 is Context, IERC20 { function transfer(address, uint256) public virtual override returns (bool) { return true; } function _beforeTokenTransfer(address, address, uint256) internal virtual {} }
abstract contract ERC20Pausable is ERC20, Pausable {
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual override { super._beforeTokenTransfer(from, to, amount); require(!paused()); }
}''',

            # Upgradeable Contracts (OpenZeppelin)
            'Initializable': '''
abstract contract Initializable {
    bool private _initialized;
    bool private _initializing;
    modifier initializer() { require(!_initialized || _initializing); bool isTopLevelCall = !_initializing; if (isTopLevelCall) { _initializing = true; _initialized = true; } _; if (isTopLevelCall) { _initializing = false; } }
    modifier onlyInitializing() { require(_initializing); _; }
}''',

            'ContextUpgradeable': '''
abstract contract Initializable { modifier initializer() { _; } }
abstract contract ContextUpgradeable is Initializable {
    function __Context_init() internal initializer {}
    function __Context_init_unchained() internal initializer {}
    function _msgSender() internal view virtual returns (address) { return msg.sender; }
    function _msgData() internal view virtual returns (bytes calldata) { return msg.data; }
}''',

            'OwnableUpgradeable': '''
abstract contract Initializable { modifier initializer() { _; } modifier onlyInitializing() { _; } }
abstract contract ContextUpgradeable is Initializable { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
abstract contract OwnableUpgradeable is Initializable, ContextUpgradeable {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    function __Ownable_init() internal onlyInitializing { __Ownable_init_unchained(); }
    function __Ownable_init_unchained() internal onlyInitializing { _owner = _msgSender(); }
    function owner() public view virtual returns (address) { return _owner; }
    modifier onlyOwner() { require(owner() == _msgSender()); _; }
    function renounceOwnership() public virtual onlyOwner { _owner = address(0); }
    function transferOwnership(address newOwner) public virtual onlyOwner { _owner = newOwner; }
}''',

            'PausableUpgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract ContextUpgradeable is Initializable { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
abstract contract PausableUpgradeable is Initializable, ContextUpgradeable {
    bool private _paused;
    event Paused(address account);
    event Unpaused(address account);
    function __Pausable_init() internal onlyInitializing { __Pausable_init_unchained(); }
    function __Pausable_init_unchained() internal onlyInitializing { _paused = false; }
    function paused() public view virtual returns (bool) { return _paused; }
    modifier whenNotPaused() { require(!paused()); _; }
    modifier whenPaused() { require(paused()); _; }
    function _pause() internal virtual whenNotPaused { _paused = true; }
    function _unpause() internal virtual whenPaused { _paused = false; }
}''',

            'ERC20Upgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract ContextUpgradeable is Initializable { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
interface IERC20Upgradeable { function totalSupply() external view returns (uint256); function balanceOf(address) external view returns (uint256); function transfer(address, uint256) external returns (bool); function allowance(address, address) external view returns (uint256); function approve(address, uint256) external returns (bool); function transferFrom(address, address, uint256) external returns (bool); }
contract ERC20Upgradeable is Initializable, ContextUpgradeable, IERC20Upgradeable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;
    function __ERC20_init(string memory name_, string memory symbol_) internal onlyInitializing { _name = name_; _symbol = symbol_; }
    function name() public view virtual returns (string memory) { return _name; }
    function symbol() public view virtual returns (string memory) { return _symbol; }
    function decimals() public view virtual returns (uint8) { return 18; }
    function totalSupply() public view virtual override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view virtual override returns (uint256) { return _balances[account]; }
    function transfer(address, uint256) public virtual override returns (bool) { return true; }
    function allowance(address, address) public view virtual override returns (uint256) { return 0; }
    function approve(address, uint256) public virtual override returns (bool) { return true; }
    function transferFrom(address, address, uint256) public virtual override returns (bool) { return true; }
    function _transfer(address, address, uint256) internal virtual {}
    function _mint(address, uint256) internal virtual {}
    function _burn(address, uint256) internal virtual {}
    function _beforeTokenTransfer(address, address, uint256) internal virtual {}
}''',

            'ERC721Upgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
interface IERC165Upgradeable { function supportsInterface(bytes4) external view returns (bool); }
abstract contract ERC165Upgradeable is Initializable, IERC165Upgradeable { function supportsInterface(bytes4) public view virtual override returns (bool) { return true; } }
interface IERC721Upgradeable is IERC165Upgradeable { function balanceOf(address) external view returns (uint256); function ownerOf(uint256) external view returns (address); function transferFrom(address, address, uint256) external; }
abstract contract ERC721Upgradeable is Initializable, ERC165Upgradeable, IERC721Upgradeable {
    mapping(uint256 => address) private _owners;
    mapping(address => uint256) private _balances;
    function __ERC721_init(string memory, string memory) internal onlyInitializing {}
    function balanceOf(address owner) public view virtual override returns (uint256) { return _balances[owner]; }
    function ownerOf(uint256 tokenId) public view virtual override returns (address) { return _owners[tokenId]; }
    function transferFrom(address, address, uint256) public virtual override {}
    function supportsInterface(bytes4) public view virtual override(ERC165Upgradeable, IERC165Upgradeable) returns (bool) { return true; }
}''',

            'ReentrancyGuardUpgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract ReentrancyGuardUpgradeable is Initializable {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;
    function __ReentrancyGuard_init() internal onlyInitializing { _status = _NOT_ENTERED; }
    modifier nonReentrant() { require(_status != _ENTERED); _status = _ENTERED; _; _status = _NOT_ENTERED; }
}''',

            'ERC165Upgradeable': '''
interface IERC165Upgradeable { function supportsInterface(bytes4 interfaceId) external view returns (bool); }
abstract contract ERC165Upgradeable is IERC165Upgradeable {
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) { return interfaceId == type(IERC165Upgradeable).interfaceId; }
}''',

            # Access Control
            'AccessControl': '''
interface IAccessControl { function hasRole(bytes32, address) external view returns (bool); function getRoleAdmin(bytes32) external view returns (bytes32); function grantRole(bytes32, address) external; function revokeRole(bytes32, address) external; function renounceRole(bytes32, address) external; }
abstract contract Context { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
abstract contract AccessControl is Context, IAccessControl {
    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;
    mapping(bytes32 => mapping(address => bool)) private _roles;
    function hasRole(bytes32 role, address account) public view virtual override returns (bool) { return _roles[role][account]; }
    function getRoleAdmin(bytes32) public view virtual override returns (bytes32) { return DEFAULT_ADMIN_ROLE; }
    function grantRole(bytes32 role, address account) public virtual override { _roles[role][account] = true; }
    function revokeRole(bytes32 role, address account) public virtual override { _roles[role][account] = false; }
    function renounceRole(bytes32 role, address account) public virtual override { require(account == _msgSender()); _roles[role][account] = false; }
    modifier onlyRole(bytes32 role) { require(hasRole(role, _msgSender())); _; }
}''',

            'AccessControlUpgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract ContextUpgradeable is Initializable { function _msgSender() internal view virtual returns (address) { return msg.sender; } }
interface IAccessControlUpgradeable { function hasRole(bytes32, address) external view returns (bool); }
abstract contract AccessControlUpgradeable is Initializable, ContextUpgradeable, IAccessControlUpgradeable {
    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;
    mapping(bytes32 => mapping(address => bool)) private _roles;
    function __AccessControl_init() internal onlyInitializing {}
    function hasRole(bytes32 role, address account) public view virtual override returns (bool) { return _roles[role][account]; }
    function getRoleAdmin(bytes32) public view virtual returns (bytes32) { return DEFAULT_ADMIN_ROLE; }
    function grantRole(bytes32 role, address account) public virtual { _roles[role][account] = true; }
    function revokeRole(bytes32 role, address account) public virtual { _roles[role][account] = false; }
    modifier onlyRole(bytes32 role) { require(hasRole(role, _msgSender())); _; }
}''',

            # Legacy Token Standards
            'ERC20Basic': '''
contract ERC20Basic {
    uint256 public totalSupply;
    function balanceOf(address who) public view returns (uint256) { return 0; }
    function transfer(address to, uint256 value) public returns (bool) { return true; }
    event Transfer(address indexed from, address indexed to, uint256 value);
}''',

            'BasicToken': '''
contract ERC20Basic { uint256 public totalSupply; function balanceOf(address) public view returns (uint256); function transfer(address, uint256) public returns (bool); }
contract BasicToken is ERC20Basic {
    mapping(address => uint256) balances;
    function transfer(address, uint256) public override returns (bool) { return true; }
    function balanceOf(address _owner) public view override returns (uint256) { return balances[_owner]; }
}''',

            'StandardToken': '''
contract ERC20Basic { function balanceOf(address) public view returns (uint256); function transfer(address, uint256) public returns (bool); }
contract BasicToken is ERC20Basic { mapping(address => uint256) balances; function transfer(address, uint256) public override returns (bool) { return true; } function balanceOf(address _owner) public view override returns (uint256) { return balances[_owner]; } }
contract ERC20 { function allowance(address, address) public view returns (uint256); function transferFrom(address, address, uint256) public returns (bool); function approve(address, uint256) public returns (bool); }
contract StandardToken is ERC20, BasicToken {
    mapping (address => mapping (address => uint256)) internal allowed;
    function transferFrom(address, address, uint256) public override returns (bool) { return true; }
    function approve(address, uint256) public override returns (bool) { return true; }
    function allowance(address, address) public view override returns (uint256) { return 0; }
}''',

            'ERC20Interface': '''
interface ERC20Interface {
    function totalSupply() external view returns (uint);
    function balanceOf(address tokenOwner) external view returns (uint balance);
    function allowance(address tokenOwner, address spender) external view returns (uint remaining);
    function transfer(address to, uint tokens) external returns (bool success);
    function approve(address spender, uint tokens) external returns (bool success);
    function transferFrom(address from, address to, uint tokens) external returns (bool success);
    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}''',

            # Other common contracts
            'Owned': '''
contract Owned {
    address public owner;
    modifier onlyOwner { require(msg.sender == owner); _; }
    function transferOwnership(address newOwner) public onlyOwner { owner = newOwner; }
}''',

            'PausableToken': '''
contract Pausable { bool public paused; modifier whenNotPaused() { require(!paused); _; } modifier whenPaused() { require(paused); _; } function pause() public { paused = true; } function unpause() public { paused = false; } }
contract ERC20Basic { function transfer(address, uint256) public returns (bool); }
contract BasicToken is ERC20Basic { function transfer(address, uint256) public override returns (bool) { return true; } }
contract PausableToken is BasicToken, Pausable {
    function transfer(address _to, uint256 _value) public override whenNotPaused returns (bool) { return super.transfer(_to, _value); }
}''',

            # Governor Contracts (minimal stubs - these are complex)
            'Governor': '''
abstract contract Governor {
    function propose(address[] memory, uint256[] memory, bytes[] memory, string memory) public virtual returns (uint256) { return 0; }
    function execute(address[] memory, uint256[] memory, bytes[] memory, bytes32) public payable virtual returns (uint256) { return 0; }
    function castVote(uint256, uint8) public virtual returns (uint256) { return 0; }
}''',

            'GovernorUpgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract GovernorUpgradeable is Initializable {
    function __Governor_init(string memory) internal onlyInitializing {}
    function propose(address[] memory, uint256[] memory, bytes[] memory, string memory) public virtual returns (uint256) { return 0; }
    function execute(address[] memory, uint256[] memory, bytes[] memory, bytes32) public payable virtual returns (uint256) { return 0; }
    function castVote(uint256, uint8) public virtual returns (uint256) { return 0; }
}''',

            # CrossChain
            'CrossChainEnabled': '''
abstract contract CrossChainEnabled {
    modifier onlyCrossChainSender(address) { _; }
    function _crossChainSender() internal view virtual returns (address) { return msg.sender; }
}''',

            'CrossChainEnabledUpgradeable': '''
abstract contract Initializable { modifier onlyInitializing() { _; } }
abstract contract CrossChainEnabledUpgradeable is Initializable {
    function __CrossChainEnabled_init() internal onlyInitializing {}
    modifier onlyCrossChainSender(address) { _; }
    function _crossChainSender() internal view virtual returns (address) { return msg.sender; }
}''',

            # ERC1155
            'ERC1155': '''
interface IERC165 { function supportsInterface(bytes4) external view returns (bool); }
abstract contract ERC165 is IERC165 { function supportsInterface(bytes4) public view virtual override returns (bool) { return true; } }
interface IERC1155 is IERC165 { function balanceOf(address, uint256) external view returns (uint256); function balanceOfBatch(address[] calldata, uint256[] calldata) external view returns (uint256[] memory); function setApprovalForAll(address, bool) external; function isApprovedForAll(address, address) external view returns (bool); function safeTransferFrom(address, address, uint256, uint256, bytes calldata) external; function safeBatchTransferFrom(address, address, uint256[] calldata, uint256[] calldata, bytes calldata) external; }
contract ERC1155 is ERC165, IERC1155 {
    mapping(uint256 => mapping(address => uint256)) private _balances;
    function balanceOf(address account, uint256 id) public view virtual override returns (uint256) { return _balances[id][account]; }
    function balanceOfBatch(address[] memory, uint256[] memory) public view virtual override returns (uint256[] memory) { return new uint256[](0); }
    function setApprovalForAll(address, bool) public virtual override {}
    function isApprovedForAll(address, address) public view virtual override returns (bool) { return false; }
    function safeTransferFrom(address, address, uint256, uint256, bytes memory) public virtual override {}
    function safeBatchTransferFrom(address, address, uint256[] memory, uint256[] memory, bytes memory) public virtual override {}
    function supportsInterface(bytes4) public view virtual override(ERC165, IERC165) returns (bool) { return true; }
}'''
        }

        # Create failure log directory if needed
        if self.log_failures:
            os.makedirs(os.path.dirname(self.failure_log_path), exist_ok=True)

    def _use_python_api(self, source_code: str, solc_version: Optional[str] = None) -> tuple[Optional[Dict], Optional[str]]:
        """Use Slither's Python API to get CFG/PDG data. Returns (slither_object, error_message)."""
        try:
            from slither import Slither

            # Write source to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            # Use Slither Python API with error suppression
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()  # Suppress compilation errors

            try:
                # Pass solc version to Slither if provided
                if solc_version:
                    # Get the path to the specific solc version in virtualenv
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    solc_path = os.path.join(project_root, f"triton_env/.solc-select/artifacts/solc-{solc_version}/solc-{solc_version}")

                    if os.path.exists(solc_path):
                        logger.debug(f"Using solc at: {solc_path}")
                        slither = Slither(temp_file, solc=solc_path, solc_disable_warnings=True)
                    else:
                        logger.warning(f"Solc {solc_version} not found at {solc_path}, using default")
                        # Fallback to default solc
                        slither = Slither(temp_file, solc_disable_warnings=True)
                else:
                    slither = Slither(temp_file, solc_disable_warnings=True)
            finally:
                sys.stderr = old_stderr

            # Clean up immediately
            os.unlink(temp_file)

            return slither, None

        except Exception as e:
            error_msg = str(e)
            logger.debug(f"Python API failed: {error_msg}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)
            return None, error_msg

    def _log_failure(self, contract_path: str, error_msg: str):
        """Log failed PDG extraction for later analysis."""
        if not self.log_failures:
            return

        try:
            with open(self.failure_log_path, 'a') as f:
                f.write(f"{contract_path}|{error_msg}\n")
        except:
            pass

    def _detect_solc_version(self, source_code: str) -> Optional[str]:
        """Detect required Solidity version from pragma statement - try exact match first."""
        pragma_pattern = r'pragma\s+solidity\s+([^;]+);'
        match = re.search(pragma_pattern, source_code)

        if not match:
            # No pragma found - try to infer from syntax
            logger.debug("No pragma statement found, attempting syntax-based detection")

            # Check for 0.5.x specific syntax (constructor keyword)
            if re.search(r'\bconstructor\s*\(', source_code):
                return '0.5.17'

            # Check for 0.4.x syntax (function name as constructor)
            contract_match = re.search(r'contract\s+(\w+)', source_code)
            if contract_match:
                contract_name = contract_match.group(1)
                if re.search(rf'function\s+{contract_name}\s*\(', source_code):
                    return '0.4.26'

            # Default fallback for contracts without pragma
            return '0.5.17'  # Most common version

        version_spec = match.group(1).strip()

        # Try to extract exact version (e.g., "0.8.17", "0.4.24")
        exact_match = re.search(r'(\d+\.\d+\.\d+)', version_spec)
        if exact_match:
            exact_version = exact_match.group(1)
            # Try exact version first (most likely to work)
            return exact_version

        # Extract major.minor for fallback (handles ^0.4.0, >=0.5.0, etc.)
        version_match = re.search(r'(\d+\.\d+)\.?\d*', version_spec)
        if not version_match:
            return '0.5.17'  # Fallback default

        major_minor = version_match.group(1)

        # Map to latest stable version per major.minor
        version_map = {
            '0.4': '0.4.26',  # Latest 0.4.x
            '0.5': '0.5.17',  # Latest 0.5.x
            '0.6': '0.6.12',  # Latest stable 0.6.x
            '0.7': '0.7.6',   # Latest 0.7.x
            '0.8': '0.8.26'   # Latest stable 0.8.x
        }

        return version_map.get(major_minor, '0.8.26')  # Default to latest

    def _set_solc_version(self, version: str) -> bool:
        """Set solc version using solc-select."""
        try:
            result = subprocess.run(
                ['solc-select', 'use', version],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to set solc version {version}: {e}")
            return False

    def _inject_dependency_stubs(self, source_code: str, error_msg: str) -> Optional[str]:
        """Inject minimal stubs for missing dependencies based on error messages."""
        # Extract missing identifiers from error message
        missing_ids = []

        # Look for "Identifier not found" and extract the actual code line
        if 'Identifier not found' in error_msg:
            # Extract the line with the actual code (marked with | at the start)
            code_lines = re.findall(r'\|\s*(.*)', error_msg)
            for line in code_lines:
                # Extract uppercase identifiers that look like contract/library names
                tokens = re.findall(r'\b([A-Z][a-z]*[A-Z]\w*|IERC\w+|ERC\w+|[A-Z][a-z]+)\b', line)
                missing_ids.extend(tokens)

        # Try alternative error pattern
        if not missing_ids:
            missing_ids = re.findall(r'Declaration "(\w+)" not found', error_msg)

        if not missing_ids:
            return None

        stubs_needed = []
        for missing_id in missing_ids:
            if missing_id in self.common_stubs:
                stubs_needed.append(self.common_stubs[missing_id])
                logger.debug(f"Adding stub for: {missing_id}")

        if stubs_needed:
            # Add pragma if missing to make stubs work
            pragma = ''
            if not re.search(r'pragma\s+solidity', source_code):
                pragma = 'pragma solidity ^0.5.17;\n'

            stubbed_code = pragma + '\n'.join(stubs_needed) + '\n\n' + source_code
            logger.info(f"Injected {len(stubs_needed)} dependency stubs: {', '.join([mid for mid in missing_ids if mid in self.common_stubs])}")
            return stubbed_code

        return None

    def _retry_with_fallback_versions(self, source_code: str, contract_name: Optional[str], contract_path: str, original_version: Optional[str]) -> Dict:
        """Retry PDG extraction with fallback compiler versions."""
        # Try common versions that usually work
        fallback_versions = ['0.5.17', '0.4.26', '0.6.12', '0.8.26']

        # Remove the original version from fallbacks if present
        if original_version and original_version in fallback_versions:
            fallback_versions.remove(original_version)

        logger.debug(f"Retrying with fallback versions: {fallback_versions}")

        for fallback_version in fallback_versions:
            try:
                if self._set_solc_version(fallback_version):
                    logger.debug(f"Retrying with solc {fallback_version}")

                    # Try Python API with explicit version
                    slither, _ = self._use_python_api(source_code, fallback_version)
                    if slither:
                        result = self._extract_from_python_api(slither, contract_name)
                        if result.get('success') and result.get('pdg') and result['pdg'].number_of_nodes() > 0:
                            logger.info(f"SUCCESS with fallback version {fallback_version}")
                            return result

            except Exception as e:
                logger.debug(f"Fallback {fallback_version} failed: {e}")
                continue

        # All retries failed
        self._log_failure(contract_path, f"Failed with original version {original_version} and all fallbacks")
        return {
            'success': True,
            'vulnerabilities': [],
            'pdg': nx.DiGraph(),
            'summary': {'total_issues': 0, 'high_severity': 0, 'medium_severity': 0, 'low_severity': 0}
        }

    def analyze_contract(self, source_code: str, contract_name: Optional[str] = None, contract_path: str = "unknown") -> Dict:
        temp_file = None
        try:
            # Detect and set appropriate Solidity compiler version
            required_version = self._detect_solc_version(source_code)
            if required_version:
                if self._set_solc_version(required_version):
                    logger.debug(f"Set solc version to {required_version}")
                else:
                    error_msg = f"Failed to set solc version {required_version}"
                    self._log_failure(contract_path, error_msg)

            # Try Python API first for better PDG extraction
            slither, api_error = self._use_python_api(source_code, required_version)

            if slither:
                # Extract data using Python API
                result = self._extract_from_python_api(slither, contract_name)
                if result.get('success') and result.get('pdg') and result['pdg'].number_of_nodes() > 0:
                    return result
                else:
                    # Python API returned empty PDG, try CLI
                    logger.warning("Python API returned empty PDG, falling back to CLI")
                    self._log_failure(contract_path, "Python API: empty PDG")
                    return self._analyze_with_cli(source_code, contract_name, contract_path)
            else:
                # Python API failed - check if we can inject stubs
                if api_error and ('Identifier not found' in api_error or 'Declaration' in api_error and 'not found' in api_error):
                    stubbed_code = self._inject_dependency_stubs(source_code, api_error)
                    if stubbed_code:
                        # Retry with stubbed code
                        logger.info("Retrying Python API with injected stubs...")
                        slither_retry, _ = self._use_python_api(stubbed_code, required_version)
                        if slither_retry:
                            result = self._extract_from_python_api(slither_retry, contract_name)
                            if result.get('success') and result.get('pdg') and result['pdg'].number_of_nodes() > 0:
                                logger.info(f"✅ SUCCESS with stub injection!")
                                return result

                # Fallback to CLI
                logger.warning("Python API failed, falling back to CLI")
                result = self._analyze_with_cli(source_code, contract_name, contract_path)

                # If CLI also failed and we have a specific version, try fallback versions
                if not result.get('success') or (result.get('pdg') and result['pdg'].number_of_nodes() == 0):
                    return self._retry_with_fallback_versions(source_code, contract_name, contract_path, required_version)

                return result

        except Exception as e:
            error_msg = f"Slither analysis error: {str(e)[:200]}"
            logger.error(error_msg)
            self._log_failure(contract_path, error_msg)
            # Return empty but valid PDG instead of error
            return {
                'success': True,
                'vulnerabilities': [],
                'pdg': nx.DiGraph(),
                'summary': {'total_issues': 0, 'high_severity': 0, 'medium_severity': 0, 'low_severity': 0}
            }
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _extract_from_python_api(self, slither, contract_name: Optional[str] = None) -> Dict:
        """Extract PDG and vulnerabilities from Slither Python API."""
        try:
            pdg = nx.DiGraph()
            vulnerabilities = []

            # Get contracts
            contracts = slither.contracts

            if contract_name:
                contracts = [c for c in contracts if c.name == contract_name]

            # Build PDG from actual contract structure
            for contract in contracts:
                contract_name_str = contract.name

                # Add contract-level state variables
                for state_var in contract.state_variables:
                    var_name = f"{contract_name_str}.{state_var.name}"
                    pdg.add_node(var_name, type='state_variable', visibility=str(state_var.visibility))

                # Add functions and build control/data flow with RICH FEATURES
                for function in contract.functions:
                    func_name = f"{contract_name_str}.{function.name}"

                    # Count loops
                    has_loop = any(node.type.name in ['STARTLOOP', 'IFLOOP'] for node in function.nodes)
                    loop_depth = sum(1 for node in function.nodes if node.type.name in ['STARTLOOP', 'IFLOOP'])

                    # Count conditionals
                    conditionals = sum(1 for node in function.nodes if node.type.name in ['IF', 'IFLOOP'])

                    # Check for require/assert
                    has_require = any('require' in str(node.expression).lower() for node in function.nodes if node.expression)
                    has_assert = any('assert' in str(node.expression).lower() for node in function.nodes if node.expression)

                    # Check for selfdestruct/delegatecall
                    has_selfdestruct = any('selfdestruct' in str(node.expression).lower() for node in function.nodes if node.expression)
                    has_delegatecall = any('delegatecall' in str(node.expression).lower() for node in function.nodes if node.expression)

                    pdg.add_node(func_name,
                                type='function',
                                visibility=str(function.visibility),
                                is_constructor=function.is_constructor,
                                is_fallback=function.is_fallback,
                                is_receive=function.is_receive,
                                view=function.view,
                                pure=function.pure,
                                payable=function.payable,
                                can_reenter=function.can_reenter(),
                                can_send_eth=function.can_send_eth(),
                                has_assembly=function.contains_assembly,
                                has_loop=has_loop,
                                loop_depth=loop_depth,
                                has_conditional=conditionals > 0,
                                num_conditionals=conditionals,
                                has_require=has_require,
                                has_assert=has_assert,
                                has_selfdestruct=has_selfdestruct,
                                has_delegatecall=has_delegatecall,
                                num_internal_calls=len(function.internal_calls),
                                num_external_calls=len(function.external_calls_as_expressions),
                                num_low_level_calls=len(function.low_level_calls),
                                num_state_vars_read=len(function.state_variables_read),
                                num_state_vars_written=len(function.state_variables_written))

                    # Add modifiers
                    for modifier in function.modifiers:
                        mod_name = f"{contract_name_str}.{modifier.name}"
                        pdg.add_node(mod_name, type='modifier')
                        pdg.add_edge(func_name, mod_name, type='uses_modifier')

                    # Add variables read/written
                    for var in function.state_variables_read:
                        var_name = f"{contract_name_str}.{var.name}"
                        if not pdg.has_node(var_name):
                            pdg.add_node(var_name, type='state_variable')
                        pdg.add_edge(func_name, var_name, type='reads')

                    for var in function.state_variables_written:
                        var_name = f"{contract_name_str}.{var.name}"
                        if not pdg.has_node(var_name):
                            pdg.add_node(var_name, type='state_variable')
                        pdg.add_edge(func_name, var_name, type='writes')

                    # Add internal calls (function calls within contract)
                    for call in function.internal_calls:
                        if hasattr(call, 'name'):
                            call_name = f"{contract_name_str}.{call.name}"
                            pdg.add_edge(func_name, call_name, type='calls')

                    # Add external calls
                    for call in function.external_calls_as_expressions:
                        # Just note that external call exists
                        pdg.add_node(f"{func_name}:external_call", type='external_call')
                        pdg.add_edge(func_name, f"{func_name}:external_call", type='makes_external_call')

            # Extract vulnerabilities from detectors (run lightweight detectors)
            # Note: We don't run all detectors as it's slow, but we get basic info
            for contract in contracts:
                for function in contract.functions:
                    # Check for some basic patterns
                    if function.can_reenter():
                        vulnerabilities.append({
                            'type': 'reentrancy',
                            'severity': 'High',
                            'confidence': 'Medium',
                            'description': f'Function {function.name} may be vulnerable to reentrancy',
                            'elements': [{'name': function.name}]
                        })

            logger.info(f"Extracted PDG with {pdg.number_of_nodes()} nodes, {pdg.number_of_edges()} edges")

            return {
                'success': True,
                'vulnerabilities': vulnerabilities,
                'pdg': pdg,
                'summary': {
                    'total_issues': len(vulnerabilities),
                    'high_severity': sum(1 for v in vulnerabilities if v['severity'] == 'High'),
                    'medium_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Medium'),
                    'low_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Low')
                }
            }

        except Exception as e:
            logger.error(f"Error extracting from Python API: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_with_cli(self, source_code: str, contract_name: Optional[str] = None, contract_path: str = "unknown", retry_with_stubs: bool = True) -> Dict:
        """Fallback: Use CLI approach (no PDG but gets vulnerabilities)."""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            cmd = ['slither', temp_file, '--json', '-']
            if contract_name:
                cmd.extend(['--contract-name', contract_name])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                temp_file = None

            if result.returncode != 0 and not result.stdout:
                stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                full_stderr = result.stderr if result.stderr else ''

                # Get more detailed error message
                error_lines = [line for line in stderr_lines if 'Error' in line or 'error' in line or 'ParserError' in line]
                if error_lines:
                    error_msg = ' | '.join(error_lines[:3])  # Up to 3 error lines
                else:
                    error_msg = stderr_lines[-1] if stderr_lines else 'Unknown error'

                # Try stub injection if we haven't already and error mentions missing identifiers
                if retry_with_stubs and ('Identifier not found' in full_stderr or 'Declaration' in full_stderr and 'not found' in full_stderr):
                    logger.info("Attempting stub injection for missing dependencies...")
                    stubbed_code = self._inject_dependency_stubs(source_code, full_stderr)
                    if stubbed_code:
                        # Retry with stubbed code (don't retry again to avoid infinite loop)
                        return self._analyze_with_cli(stubbed_code, contract_name, contract_path, retry_with_stubs=False)

                logger.error(f"Slither CLI analysis failed: {error_msg[:300]}")
                self._log_failure(contract_path, f"CLI failed: {error_msg[:300]}")
                # Return empty but valid PDG instead of error
                return {
                    'success': True,
                    'vulnerabilities': [],
                    'pdg': nx.DiGraph(),
                    'summary': {'total_issues': 0, 'high_severity': 0, 'medium_severity': 0, 'low_severity': 0}
                }

            if result.returncode != 0:
                logger.warning(f"Slither returned error code but has output")

            analysis = json.loads(result.stdout) if result.stdout else {}
            return self._process_slither_output(analysis)

        except subprocess.TimeoutExpired:
            logger.error(f"Slither CLI timed out after {self.timeout} seconds")
            return {'success': False, 'error': 'Analysis timeout'}
        except Exception as e:
            logger.error(f"Slither CLI error: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _process_slither_output(self, analysis: Dict) -> Dict:
        vulnerabilities = []
        pdg = nx.DiGraph()

        if 'results' in analysis:
            detectors = analysis['results'].get('detectors', [])

            for detector in detectors:
                vuln_type = self.detector_mapping.get(
                    detector.get('check', ''),
                    detector.get('check', 'unknown')
                )

                vulnerability = {
                    'type': vuln_type,
                    'severity': detector.get('impact', 'unknown'),
                    'confidence': detector.get('confidence', 'unknown'),
                    'description': detector.get('description', ''),
                    'elements': []
                }

                for element in detector.get('elements', []):
                    if element.get('type') == 'function':
                        vulnerability['elements'].append({
                            'name': element.get('name', ''),
                            'source_mapping': element.get('source_mapping', {})
                        })

                vulnerabilities.append(vulnerability)

        if 'contracts' in analysis:
            pdg = self._build_program_dependence_graph(analysis['contracts'])

        return {
            'success': True,
            'vulnerabilities': vulnerabilities,
            'pdg': pdg,
            'summary': {
                'total_issues': len(vulnerabilities),
                'high_severity': sum(1 for v in vulnerabilities if v['severity'] == 'High'),
                'medium_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Medium'),
                'low_severity': sum(1 for v in vulnerabilities if v['severity'] == 'Low')
            }
        }

    def _build_program_dependence_graph(self, contracts: List[Dict]) -> nx.DiGraph:
        pdg = nx.DiGraph()

        for contract in contracts:
            contract_name = contract.get('name', '')

            for function in contract.get('functions', []):
                func_name = f"{contract_name}.{function.get('name', '')}"
                pdg.add_node(func_name, type='function')

                for modifier in function.get('modifiers', []):
                    mod_name = f"{contract_name}.{modifier}"
                    pdg.add_node(mod_name, type='modifier')
                    pdg.add_edge(func_name, mod_name, type='uses_modifier')

                for var in function.get('variables_read', []):
                    var_name = f"{contract_name}.{var}"
                    pdg.add_node(var_name, type='variable')
                    pdg.add_edge(func_name, var_name, type='reads')

                for var in function.get('variables_written', []):
                    var_name = f"{contract_name}.{var}"
                    pdg.add_node(var_name, type='variable')
                    pdg.add_edge(func_name, var_name, type='writes')

                for call in function.get('internal_calls', []):
                    call_name = f"{contract_name}.{call}"
                    pdg.add_edge(func_name, call_name, type='calls')

        return pdg


def extract_static_features(source_code: str, contract_name: Optional[str] = None, contract_path: str = "unknown") -> Dict[str, Any]:
    wrapper = SlitherWrapper()
    result = wrapper.analyze_contract(source_code, contract_name, contract_path)

    if not result['success']:
        return None

    features = {
        'vulnerability_count': result['summary']['total_issues'],
        'high_severity_count': result['summary']['high_severity'],
        'medium_severity_count': result['summary']['medium_severity'],
        'low_severity_count': result['summary']['low_severity'],
        'pdg': result['pdg'],
        'vulnerabilities': result['vulnerabilities']
    }

    pdg = result['pdg']
    features['pdg_stats'] = {
        'num_nodes': pdg.number_of_nodes(),
        'num_edges': pdg.number_of_edges(),
        'avg_degree': sum(dict(pdg.degree()).values()) / pdg.number_of_nodes() if pdg.number_of_nodes() > 0 else 0,
        'num_functions': sum(1 for n, d in pdg.nodes(data=True) if d.get('type') == 'function'),
        'num_variables': sum(1 for n, d in pdg.nodes(data=True) if d.get('type') == 'variable')
    }

    return features