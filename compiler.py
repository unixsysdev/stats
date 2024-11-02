from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Any

# Extended Token types
class TokenType(Enum):
    # Keywords
    PRINT = "PRINT"
    IF = "IF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    
    # Data types
    STRING = "STRING"
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    
    # Operators
    EQUALS = "EQUALS"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    
    # Comparisons
    GREATER = "GREATER"
    LESS = "LESS"
    EQUAL_EQUAL = "EQUAL_EQUAL"
    
    # Syntax
    SEMICOLON = "SEMICOLON"
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    LEFT_BRACE = "LEFT_BRACE"
    RIGHT_BRACE = "RIGHT_BRACE"
    
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.current_char = self.source[0] if source else None
        
        # Define keywords
        self.keywords = {
            'print': TokenType.PRINT,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
        }

    def advance(self):
        self.position += 1
        if self.position >= len(self.source):
            self.current_char = None
        else:
            if self.current_char == '\n':
                self.line += 1
            self.current_char = self.source[self.position]

    def peek(self) -> Optional[str]:
        peek_pos = self.position + 1
        return self.source[peek_pos] if peek_pos < len(self.source) else None

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def read_string(self) -> str:
        result = ""
        self.advance()  # Skip opening quote
        while self.current_char and self.current_char != '"':
            result += self.current_char
            self.advance()
        self.advance()  # Skip closing quote
        return result

    def read_number(self) -> float:
        result = ""
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return float(result)

    def read_identifier(self) -> str:
        result = ""
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result

    def get_next_token(self) -> Token:
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '"':
                return Token(TokenType.STRING, self.read_string(), self.line)

            if self.current_char.isdigit():
                return Token(TokenType.NUMBER, str(self.read_number()), self.line)

            if self.current_char.isalpha():
                identifier = self.read_identifier()
                token_type = self.keywords.get(identifier.lower(), TokenType.IDENTIFIER)
                return Token(token_type, identifier, self.line)

            if self.current_char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.EQUAL_EQUAL, '==', self.line)

            if self.current_char == '=':
                self.advance()
                return Token(TokenType.EQUALS, '=', self.line)

            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS, '+', self.line)

            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS, '-', self.line)

            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MULTIPLY, '*', self.line)

            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE, '/', self.line)

            if self.current_char == '>':
                self.advance()
                return Token(TokenType.GREATER, '>', self.line)

            if self.current_char == '<':
                self.advance()
                return Token(TokenType.LESS, '<', self.line)

            if self.current_char == ';':
                self.advance()
                return Token(TokenType.SEMICOLON, ';', self.line)

            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LEFT_PAREN, '(', self.line)

            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RIGHT_PAREN, ')', self.line)

            if self.current_char == '{':
                self.advance()
                return Token(TokenType.LEFT_BRACE, '{', self.line)

            if self.current_char == '}':
                self.advance()
                return Token(TokenType.RIGHT_BRACE, '}', self.line)

            raise SyntaxError(f"Invalid character '{self.current_char}' at line {self.line}")

        return Token(TokenType.EOF, "", self.line)


# First, let's define our AST nodes
@dataclass
class NumberNode:
    value: float

@dataclass
class StringNode:
    value: str

@dataclass
class VariableNode:
    name: str

@dataclass
class BinaryOpNode:
    left: Any
    operator: str
    right: Any

@dataclass
class AssignmentNode:
    name: str
    value: Any

@dataclass
class PrintNode:
    expression: Any

@dataclass
class IfNode:
    condition: Any
    if_block: List[Any]
    else_block: Optional[List[Any]]

@dataclass
class WhileNode:
    condition: Any
    block: List[Any]

@dataclass
class ComparisonNode:
    left: Any
    operator: str
    right: Any

# Parser now creates AST nodes instead of executing directly
class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token.type}")

    def parse_factor(self) -> Union[NumberNode, StringNode, VariableNode]:
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberNode(float(token.value))
        
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return StringNode(token.value)
        
        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            return VariableNode(token.value)
        
        elif token.type == TokenType.LEFT_PAREN:
            self.eat(TokenType.LEFT_PAREN)
            result = self.parse_expression()
            self.eat(TokenType.RIGHT_PAREN)
            return result
        
        raise SyntaxError(f"Unexpected token {token.type}")

    def parse_term(self) -> Union[NumberNode, BinaryOpNode]:
        node = self.parse_factor()
        
        while self.current_token.type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            token = self.current_token
            if token.type == TokenType.MULTIPLY:
                self.eat(TokenType.MULTIPLY)
                node = BinaryOpNode(node, '*', self.parse_factor())
            elif token.type == TokenType.DIVIDE:
                self.eat(TokenType.DIVIDE)
                node = BinaryOpNode(node, '/', self.parse_factor())
                
        return node

    def parse_expression(self) -> Union[NumberNode, BinaryOpNode]:
        node = self.parse_term()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
                node = BinaryOpNode(node, '+', self.parse_term())
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)
                node = BinaryOpNode(node, '-', self.parse_term())
                
        return node

    def parse_condition(self) -> ComparisonNode:
        left = self.parse_expression()
        
        if self.current_token.type in (TokenType.GREATER, TokenType.LESS, TokenType.EQUAL_EQUAL):
            operator = self.current_token.type
            self.eat(operator)
            right = self.parse_expression()
            return ComparisonNode(left, operator.value, right)
                
        raise SyntaxError("Expected comparison operator")

    def parse_block(self) -> List[Any]:
        statements = []
        self.eat(TokenType.LEFT_BRACE)
        while self.current_token.type != TokenType.RIGHT_BRACE:
            statements.append(self.parse_statement())
        self.eat(TokenType.RIGHT_BRACE)
        return statements

    def parse_statement(self) -> Union[PrintNode, AssignmentNode, IfNode, WhileNode]:
        if self.current_token.type == TokenType.PRINT:
            self.eat(TokenType.PRINT)
            expr = self.parse_expression()
            self.eat(TokenType.SEMICOLON)
            return PrintNode(expr)
            
        elif self.current_token.type == TokenType.IF:
            self.eat(TokenType.IF)
            self.eat(TokenType.LEFT_PAREN)
            condition = self.parse_condition()
            self.eat(TokenType.RIGHT_PAREN)
            
            if_block = self.parse_block()
            
            else_block = None
            if self.current_token.type == TokenType.ELSE:
                self.eat(TokenType.ELSE)
                else_block = self.parse_block()
            
            return IfNode(condition, if_block, else_block)
                    
        elif self.current_token.type == TokenType.WHILE:
            self.eat(TokenType.WHILE)
            self.eat(TokenType.LEFT_PAREN)
            condition = self.parse_condition()
            self.eat(TokenType.RIGHT_PAREN)
            block = self.parse_block()
            return WhileNode(condition, block)
                
        elif self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.eat(TokenType.IDENTIFIER)
            self.eat(TokenType.EQUALS)
            value = self.parse_expression()
            self.eat(TokenType.SEMICOLON)
            return AssignmentNode(name, value)

    def parse(self) -> List[Any]:
        statements = []
        while self.current_token.type != TokenType.EOF:
            statements.append(self.parse_statement())
        return statements

# New Interpreter class that executes the AST
class Interpreter:
    def __init__(self):
        self.variables: Dict[str, Union[float, str]] = {}

    def visit_NumberNode(self, node: NumberNode) -> float:
        return node.value

    def visit_StringNode(self, node: StringNode) -> str:
        return node.value

    def visit_VariableNode(self, node: VariableNode) -> Union[float, str]:
        if node.name not in self.variables:
            raise NameError(f"Variable '{node.name}' not defined")
        return self.variables[node.name]

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if node.operator == '+':
            return float(left) + float(right)
        elif node.operator == '-':
            return float(left) - float(right)
        elif node.operator == '*':
            return float(left) * float(right)
        elif node.operator == '/':
            if float(right) == 0:
                raise ZeroDivisionError("Division by zero")
            return float(left) / float(right)
        
        raise ValueError(f"Unknown operator {node.operator}")

    def visit_AssignmentNode(self, node: AssignmentNode) -> None:
        self.variables[node.name] = self.visit(node.value)

    def visit_PrintNode(self, node: PrintNode) -> None:
        value = self.visit(node.expression)
        print(value)

    def visit_ComparisonNode(self, node: ComparisonNode) -> bool:
        left = float(self.visit(node.left))
        right = float(self.visit(node.right))
        
        if node.operator == 'GREATER':
            return left > right
        elif node.operator == 'LESS':
            return left < right
        elif node.operator == 'EQUAL_EQUAL':
            return left == right
        
        raise ValueError(f"Unknown comparison operator {node.operator}")

    def visit_IfNode(self, node: IfNode) -> None:
        if self.visit(node.condition):
            for statement in node.if_block:
                self.visit(statement)
        elif node.else_block:
            for statement in node.else_block:
                self.visit(statement)

    def visit_WhileNode(self, node: WhileNode) -> None:
        while self.visit(node.condition):
            for statement in node.block:
                self.visit(statement)

    def visit(self, node: Any) -> Any:
        # Dispatch to the appropriate visit method based on node type
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name)
        return method(node)

def run_program(source: str):
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    interpreter = Interpreter()
    
    for statement in ast:
        interpreter.visit(statement)

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Any, Tuple
import time
from functools import lru_cache

# Bytecode operations
class OpCode(Enum):
    LOAD_CONST = auto()
    LOAD_VAR = auto()
    STORE_VAR = auto()
    BINARY_ADD = auto()
    BINARY_SUB = auto()
    BINARY_MUL = auto()
    BINARY_DIV = auto()
    COMPARE_GT = auto()
    COMPARE_LT = auto()
    COMPARE_EQ = auto()
    PRINT = auto()
    JUMP_IF_FALSE = auto()
    JUMP = auto()

@dataclass
class Instruction:
    opcode: OpCode
    operand: Any = None

class BytecodeCompiler:
    def __init__(self):
        self.constants: List[Any] = []
        self.instructions: List[Instruction] = []
    
    def add_constant(self, value: Any) -> int:
        # Reuse existing constant if possible
        for i, const in enumerate(self.constants):
            if const == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1
    
    def emit(self, opcode: OpCode, operand: Any = None):
        self.instructions.append(Instruction(opcode, operand))
        return len(self.instructions) - 1

    def compile_node(self, node: Any) -> None:
        method_name = f'compile_{type(node).__name__}'
        method = getattr(self, method_name)
        return method(node)

    def compile_NumberNode(self, node: NumberNode) -> None:
        const_idx = self.add_constant(node.value)
        self.emit(OpCode.LOAD_CONST, const_idx)

    def compile_StringNode(self, node: StringNode) -> None:
        const_idx = self.add_constant(node.value)
        self.emit(OpCode.LOAD_CONST, const_idx)

    def compile_VariableNode(self, node: VariableNode) -> None:
        self.emit(OpCode.LOAD_VAR, node.name)

    def compile_BinaryOpNode(self, node: BinaryOpNode) -> None:
        # Constant folding optimization
        if isinstance(node.left, NumberNode) and isinstance(node.right, NumberNode):
            # Compute the result at compile time
            left_val = node.left.value
            right_val = node.right.value
            result = None
            
            if node.operator == '+':
                result = left_val + right_val
            elif node.operator == '-':
                result = left_val - right_val
            elif node.operator == '*':
                result = left_val * right_val
            elif node.operator == '/' and right_val != 0:
                result = left_val / right_val
                
            if result is not None:
                const_idx = self.add_constant(result)
                self.emit(OpCode.LOAD_CONST, const_idx)
                return

        # If we can't fold constants, compile normally
        self.compile_node(node.left)
        self.compile_node(node.right)
        
        if node.operator == '+':
            self.emit(OpCode.BINARY_ADD)
        elif node.operator == '-':
            self.emit(OpCode.BINARY_SUB)
        elif node.operator == '*':
            self.emit(OpCode.BINARY_MUL)
        elif node.operator == '/':
            self.emit(OpCode.BINARY_DIV)

    def compile_AssignmentNode(self, node: AssignmentNode) -> None:
        self.compile_node(node.value)
        self.emit(OpCode.STORE_VAR, node.name)

    def compile_PrintNode(self, node: PrintNode) -> None:
        self.compile_node(node.expression)
        self.emit(OpCode.PRINT)

    def compile_ComparisonNode(self, node: ComparisonNode) -> None:
        self.compile_node(node.left)
        self.compile_node(node.right)
        
        if node.operator == 'GREATER':
            self.emit(OpCode.COMPARE_GT)
        elif node.operator == 'LESS':
            self.emit(OpCode.COMPARE_LT)
        elif node.operator == 'EQUAL_EQUAL':
            self.emit(OpCode.COMPARE_EQ)

    def compile_IfNode(self, node: IfNode) -> None:
        self.compile_node(node.condition)
        
        # Emit jump placeholder
        jump_if_false_idx = self.emit(OpCode.JUMP_IF_FALSE, 0)
        
        # Compile if block
        for stmt in node.if_block:
            self.compile_node(stmt)
            
        # Emit jump placeholder for else block
        jump_idx = None
        if node.else_block:
            jump_idx = self.emit(OpCode.JUMP, 0)
            
        # Patch jump_if_false
        self.instructions[jump_if_false_idx].operand = len(self.instructions)
        
        # Compile else block if it exists
        if node.else_block:
            for stmt in node.else_block:
                self.compile_node(stmt)
            # Patch jump
            self.instructions[jump_idx].operand = len(self.instructions)

    def compile_WhileNode(self, node: WhileNode) -> None:
        loop_start = len(self.instructions)
        
        # Compile condition
        self.compile_node(node.condition)
        
        # Emit jump placeholder
        jump_if_false_idx = self.emit(OpCode.JUMP_IF_FALSE, 0)
        
        # Compile loop body
        for stmt in node.block:
            self.compile_node(stmt)
            
        # Jump back to start
        self.emit(OpCode.JUMP, loop_start)
        
        # Patch jump_if_false
        self.instructions[jump_if_false_idx].operand = len(self.instructions)

class OptimizedInterpreter:
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.stack: List[Any] = []
        
    @lru_cache(maxsize=1024)
    def _binary_op(self, op: str, a: float, b: float) -> float:
        """Cache results of binary operations for common values"""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            return a / b
        raise ValueError(f"Unknown operator {op}")

    def execute(self, compiler: BytecodeCompiler) -> None:
        ip = 0  # Instruction pointer
        while ip < len(compiler.instructions):
            instruction = compiler.instructions[ip]
            
            if instruction.opcode == OpCode.LOAD_CONST:
                self.stack.append(compiler.constants[instruction.operand])
            
            elif instruction.opcode == OpCode.LOAD_VAR:
                if instruction.operand not in self.variables:
                    raise NameError(f"Variable '{instruction.operand}' not defined")
                self.stack.append(self.variables[instruction.operand])
            
            elif instruction.opcode == OpCode.STORE_VAR:
                self.variables[instruction.operand] = self.stack.pop()
            
            elif instruction.opcode in (OpCode.BINARY_ADD, OpCode.BINARY_SUB, 
                                     OpCode.BINARY_MUL, OpCode.BINARY_DIV):
                right = float(self.stack.pop())
                left = float(self.stack.pop())
                
                if instruction.opcode == OpCode.BINARY_ADD:
                    result = self._binary_op('+', left, right)
                elif instruction.opcode == OpCode.BINARY_SUB:
                    result = self._binary_op('-', left, right)
                elif instruction.opcode == OpCode.BINARY_MUL:
                    result = self._binary_op('*', left, right)
                elif instruction.opcode == OpCode.BINARY_DIV:
                    if right == 0:
                        raise ZeroDivisionError("Division by zero")
                    result = self._binary_op('/', left, right)
                
                self.stack.append(result)
            
            elif instruction.opcode in (OpCode.COMPARE_GT, OpCode.COMPARE_LT, 
                                     OpCode.COMPARE_EQ):
                right = float(self.stack.pop())
                left = float(self.stack.pop())
                
                if instruction.opcode == OpCode.COMPARE_GT:
                    self.stack.append(left > right)
                elif instruction.opcode == OpCode.COMPARE_LT:
                    self.stack.append(left < right)
                elif instruction.opcode == OpCode.COMPARE_EQ:
                    self.stack.append(left == right)
            
            elif instruction.opcode == OpCode.PRINT:
                print(self.stack.pop())
            
            elif instruction.opcode == OpCode.JUMP_IF_FALSE:
                condition = self.stack.pop()
                if not condition:
                    ip = instruction.operand
                    continue
            
            elif instruction.opcode == OpCode.JUMP:
                ip = instruction.operand
                continue
            
            ip += 1

def run_optimized_program(source: str) -> float:
    start_time = time.time()
    
    # Parse to AST
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    
    # Compile to bytecode
    compiler = BytecodeCompiler()
    for node in ast:
        compiler.compile_node(node)
    
    # Execute bytecode
    interpreter = OptimizedInterpreter()
    interpreter.execute(compiler)
    
    end_time = time.time()
    return end_time - start_time

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import struct
import os
from pathlib import Path

# ELF Constants
EI_NIDENT = 16
ELFCLASS64 = 2
ELFDATA2LSB = 1
EV_CURRENT = 1
ET_EXEC = 2
EM_X86_64 = 62
PT_LOAD = 1

@dataclass
class ElfHeader:
    """ELF64 Header structure"""
    e_ident: bytes = bytes([
        0x7f, ord('E'), ord('L'), ord('F'),  # Magic number
        ELFCLASS64,                          # 64-bit
        ELFDATA2LSB,                         # Little-endian
        EV_CURRENT,                          # Current version
        0,                                   # System V ABI
        0,                                   # ABI Version
        *[0] * (EI_NIDENT - 9)              # Padding
    ])
    e_type: int = ET_EXEC
    e_machine: int = EM_X86_64
    e_version: int = EV_CURRENT
    e_entry: int = 0x400000                 # Entry point
    e_phoff: int = 64                       # Program header offset
    e_shoff: int = 0                        # Section header offset
    e_flags: int = 0
    e_ehsize: int = 64                      # ELF header size
    e_phentsize: int = 56                   # Program header entry size
    e_phnum: int = 1                        # Number of program headers
    e_shentsize: int = 64                   # Section header entry size
    e_shnum: int = 0                        # Number of section headers
    e_shstrndx: int = 0                     # Section header string table index

    def pack(self) -> bytes:
        return struct.pack(
            '<16sHHIQQQIHHHHHH',
            self.e_ident,
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.e_phoff,
            self.e_shoff,
            self.e_flags,
            self.e_ehsize,
            self.e_phentsize,
            self.e_phnum,
            self.e_shentsize,
            self.e_shnum,
            self.e_shstrndx
        )

@dataclass
class ProgramHeader:
    """Program Header structure"""
    p_type: int = PT_LOAD
    p_flags: int = 5                        # Read + Execute
    p_offset: int = 0
    p_vaddr: int = 0x400000                # Virtual address
    p_paddr: int = 0x400000                # Physical address
    p_filesz: int = 0                       # File size
    p_memsz: int = 0                        # Memory size
    p_align: int = 0x1000                   # Alignment

    def pack(self) -> bytes:
        return struct.pack(
            '<IIQQQQQQ',
            self.p_type,
            self.p_flags,
            self.p_offset,
            self.p_vaddr,
            self.p_paddr,
            self.p_filesz,
            self.p_memsz,
            self.p_align
        )
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import struct
import os

# Previous ELF-related code remains the same...

class X86_64Assembler:
    def __init__(self):
        self.code = bytearray()
        self.string_pool = []
        self.data_section = bytearray()
        self.data_offset = 0
        self.string_offsets = {}

    def calculate_string_offset(self, s: str) -> int:
        """Calculate where the string will be in memory"""
        offset = len(self.code)
        for existing in self.string_pool:
            offset += len(existing) + 1  # +1 for null terminator
        return offset

    def add_string(self, s: str) -> int:
        if s in self.string_offsets:
            return self.string_offsets[s]

        offset = self.calculate_string_offset(s)
        self.string_offsets[s] = offset
        self.string_pool.append(s.encode())
        return offset

    def mov_rax_imm(self, imm: int):
        """mov rax, imm"""
        self.code.extend([0x48, 0xb8])
        self.code.extend(struct.pack('<Q', imm))

    def mov_rdi_imm(self, imm: int):
        """mov rdi, imm"""
        self.code.extend([0x48, 0xbf])
        self.code.extend(struct.pack('<Q', imm))

    def mov_rsi_imm(self, imm: int):
        """mov rsi, imm"""
        self.code.extend([0x48, 0xbe])
        self.code.extend(struct.pack('<Q', imm))

    def mov_rdx_imm(self, imm: int):
        """mov rdx, imm"""
        self.code.extend([0x48, 0xba])
        self.code.extend(struct.pack('<Q', imm))

    def syscall(self):
        """syscall"""
        self.code.extend([0x0f, 0x05])

    def store_immediate(self, value: int):
        """Store immediate value in data section"""
        offset = self.data_offset
        self.data_section.extend(struct.pack('<Q', value))
        self.data_offset += 8
        return offset

    def __init__(self):
        self.code = bytearray()
        self.string_pool = []
        self.data_section = bytearray()
        self.data_offset = 0
        self.string_offsets = {}

    def allocate_data(self, size: int) -> int:
        """Allocate space in data section and return offset"""
        offset = self.data_offset
        self.data_section.extend([0] * size)
        self.data_offset += size
        return offset

    def mov_rax_mem(self, offset: int):
        """mov rax, [rip + offset]"""
        rel_offset = 0x600000 + offset - (0x400000 + len(self.code) + 7)
        self.code.extend([0x48, 0x8b, 0x05])
        self.code.extend(struct.pack('<i', rel_offset))

    def mov_mem_rax(self, offset: int):
        """mov [rip + offset], rax"""
        rel_offset = 0x600000 + offset - (0x400000 + len(self.code) + 7)
        self.code.extend([0x48, 0x89, 0x05])
        self.code.extend(struct.pack('<i', rel_offset))

    def add_rax_imm(self, imm: int):
        """add rax, imm"""
        if -128 <= imm <= 127:
            self.code.extend([0x48, 0x83, 0xc0, imm & 0xff])
        else:
            self.code.extend([0x48, 0x05])
            self.code.extend(struct.pack('<I', imm))

    def sub_rax_imm(self, imm: int):
        """sub rax, imm"""
        if -128 <= imm <= 127:
            self.code.extend([0x48, 0x83, 0xe8, imm & 0xff])
        else:
            self.code.extend([0x48, 0x2d])
            self.code.extend(struct.pack('<I', imm))

    def mul_rax_imm(self, imm: int):
        """imul rax, imm"""
        if -128 <= imm <= 127:
            self.code.extend([0x48, 0x6b, 0xc0, imm & 0xff])
        else:
            self.code.extend([0x48, 0x69, 0xc0])
            self.code.extend(struct.pack('<I', imm))

class NativeCompiler:
    def __init__(self):
        self.assembler = X86_64Assembler()
        self.variables: Dict[str, int] = {}  # Variable name to memory offset
        self.next_var_offset = 0

    def get_variable_offset(self, name: str) -> int:
        """Get or create memory offset for variable"""
        if name not in self.variables:
            offset = self.assembler.allocate_data(8)  # Allocate 8 bytes for each variable
            self.variables[name] = offset
            self.next_var_offset = offset + 8
        return self.variables[name]

    def compile_node(self, node: Any) -> None:
        method_name = f'compile_{type(node).__name__}'
        method = getattr(self, method_name)
        return method(node)

    def compile_NumberNode(self, node: NumberNode) -> None:
        self.assembler.mov_rax_imm(int(node.value))

    def compile_StringNode(self, node: StringNode) -> None:
        offset = self.assembler.add_string(node.value)
        # Calculate string address relative to instruction pointer
        self.assembler.mov_rax_imm(0x400000 + len(self.assembler.code) + offset)

    def compile_VariableNode(self, node: VariableNode) -> None:
        if node.name not in self.variables:
            raise NameError(f"Variable '{node.name}' not defined")
        offset = self.variables[node.name]
        self.assembler.mov_rax_mem(offset)

    def compile_AssignmentNode(self, node: AssignmentNode) -> None:
        # Compile the value
        self.compile_node(node.value)
        # Store it in variable's memory location
        offset = self.get_variable_offset(node.name)
        self.assembler.mov_mem_rax(offset)

    def compile_BinaryOpNode(self, node: BinaryOpNode) -> None:
        # Compile right operand first
        self.compile_node(node.right)
        # Save result
        self.assembler.mov_rdi_rax()
        # Compile left operand
        self.compile_node(node.left)
        
        # Perform operation
        if node.operator == '+':
            self.assembler.add_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 0)
        elif node.operator == '-':
            self.assembler.sub_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 0)
        elif node.operator == '*':
            self.assembler.mul_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 1)

    def compile_PrintNode(self, node: PrintNode) -> None:
        # Compile expression to print
        self.compile_node(node.expression)
        
        # Print syscall (write)
        self.assembler.mov_rdi_imm(1)  # stdout
        self.assembler.mov_rdi_rax()   # buffer
        self.assembler.mov_rax_imm(1)  # sys_write
        self.assembler.syscall()

    def compile_StringNode(self, node: StringNode) -> None:
        # Get string offset and length
        offset = self.assembler.add_string(node.value)
        length = len(node.value)

        # sys_write(1, str_addr, len)
        self.assembler.mov_rax_imm(1)            # sys_write
        self.assembler.mov_rdi_imm(1)            # stdout
        self.assembler.mov_rsi_imm(0x400000 + offset)  # string address
        self.assembler.mov_rdx_imm(length)       # string length
        self.assembler.syscall()

    def compile_NumberNode(self, node: NumberNode) -> None:
        # Store number in data section and load its address
        offset = self.assembler.store_immediate(int(node.value))
        # Load value into rax
        self.assembler.mov_rax_imm(int(node.value))

    def compile_PrintNode(self, node: PrintNode) -> None:
        # Compile the expression to print
        self.compile_node(node.expression)

        if isinstance(node.expression, StringNode):
            # String printing is handled in compile_StringNode
            pass
        else:
            # Convert number to string and print
            # For now, just print the raw value
            self.assembler.mov_rdi_imm(1)    # stdout
            self.assembler.mov_rsi_imm(0x600000)  # Use data section
            self.assembler.mov_rdx_imm(8)    # 8 bytes for number
            self.assembler.syscall()

    def compile_program(self, ast: List[Any]) -> bytes:
        # Generate code
        for node in ast:
            self.compile_node(node)

        # Add exit syscall
        self.assembler.mov_rax_imm(60)  # sys_exit
        self.assembler.mov_rdi_imm(0)   # status 0
        self.assembler.syscall()

        # Create ELF file
        text_size = len(self.assembler.code)
        strings_size = sum(len(s) + 1 for s in self.assembler.string_pool)
        data_size = len(self.assembler.data_section)

        # Headers
        elf_header = ElfHeader()

        # Single segment containing everything
        program_header = ProgramHeader(
            p_type=1,  # PT_LOAD
            p_flags=7,  # Read + Write + Execute
            p_offset=0,
            p_vaddr=0x400000,
            p_paddr=0x400000,
            p_filesz=0x1000,  # Fixed 4K page size
            p_memsz=0x1000,
            p_align=0x1000
        )

        # Combine everything
        result = bytearray()
        result.extend(elf_header.pack())
        result.extend(program_header.pack())

        # Add code
        result.extend(self.assembler.code)

        # Add strings (null-terminated)
        for s in self.assembler.string_pool:
            result.extend(s + b'\0')

        # Add data section
        result.extend(self.assembler.data_section)

        # Pad to page size
        padding_size = 0x1000 - len(result)
        if padding_size > 0:
            result.extend(bytes([0] * padding_size))

        return bytes(result)

    def compile_VariableNode(self, node: VariableNode) -> None:
        """Load variable value into rax"""
        if node.name not in self.variables:
            raise NameError(f"Variable '{node.name}' not defined")
        offset = self.variables[node.name]
        self.assembler.mov_rax_mem(offset)

    def compile_AssignmentNode(self, node: AssignmentNode) -> None:
        """Store value in variable"""
        # First compile the value
        self.compile_node(node.value)
        # Store rax in variable's memory location
        offset = self.get_variable_offset(node.name)
        self.assembler.mov_mem_rax(offset)

    def compile_BinaryOpNode(self, node: BinaryOpNode) -> None:
        """Compile binary operation (a op b)"""
        # First get value of right operand
        self.compile_node(node.right)
        # Save it to memory temporarily
        temp_offset = self.assembler.allocate_data(8)
        self.assembler.mov_mem_rax(temp_offset)
        
        # Get value of left operand
        self.compile_node(node.left)
        
        # Now perform the operation
        if node.operator == '+':
            # Load right operand and add
            self.assembler.mov_mem_rax(temp_offset)
            self.assembler.add_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 0)
        elif node.operator == '-':
            # Load right operand and subtract
            self.assembler.mov_mem_rax(temp_offset)
            self.assembler.sub_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 0)
        elif node.operator == '*':
            # Load right operand and multiply
            self.assembler.mov_mem_rax(temp_offset)
            self.assembler.mul_rax_imm(int(node.right.value) if isinstance(node.right, NumberNode) else 1)
        else:
            raise ValueError(f"Unknown operator: {node.operator}")

    def compile_PrintNode(self, node: PrintNode) -> None:
        """Print value to stdout"""
        # Compile expression to print
        self.compile_node(node.expression)
        
        if isinstance(node.expression, StringNode):
            # String printing handled in compile_StringNode
            pass
        else:
            # For numbers: convert to string
            # For now just print the raw bytes
            self.assembler.mov_rdi_imm(1)  # stdout
            self.assembler.mov_rsi_imm(0x600000)  # data section
            self.assembler.mov_rdx_imm(8)  # size
            self.assembler.mov_rax_imm(1)  # sys_write
            self.assembler.syscall()

    def compile_program(self, ast: List[Any]) -> bytes:
        # Add space for numeric format buffer
        self.num_format_offset = self.assembler.allocate_data(32)
        
        # Compile code
        for node in ast:
            self.compile_node(node)
        
        # Exit
        self.assembler.mov_rax_imm(60)  # sys_exit
        self.assembler.mov_rdi_imm(0)   # status code
        self.assembler.syscall()
        
        # Create final executable
        text_size = len(self.assembler.code)
        data_size = len(self.assembler.data_section)
        
        # Single loadable segment
        program_header = ProgramHeader(
            p_type=1,       # PT_LOAD
            p_flags=7,      # Read + Write + Execute
            p_offset=0,
            p_vaddr=0x400000,
            p_paddr=0x400000,
            p_filesz=0x1000,
            p_memsz=0x1000,
            p_align=0x1000
        )
        
        # Build executable
        result = bytearray()
        result.extend(ElfHeader().pack())
        result.extend(program_header.pack())
        result.extend(self.assembler.code)
        result.extend(self.assembler.data_section)
        
        # Pad to page size
        padding_size = 0x1000 - len(result)
        if padding_size > 0:
            result.extend(bytes([0] * padding_size))
            
        return bytes(result)

def generate_assembly(code: bytearray) -> str:
    """Convert machine code to AT&T syntax assembly for debugging"""
    assembly = []
    pos = 0
    while pos < len(code):
        if code[pos:pos+2] == bytes([0x48, 0xb8]):  # mov rax, imm64
            value = struct.unpack('<Q', code[pos+2:pos+10])[0]
            assembly.append(f"    movabs ${value:#x}, %rax")
            pos += 10
        elif code[pos:pos+2] == bytes([0x48, 0xbf]):  # mov rdi, imm64
            value = struct.unpack('<Q', code[pos+2:pos+10])[0]
            assembly.append(f"    movabs ${value:#x}, %rdi")
            pos += 10
        elif code[pos:pos+3] == bytes([0x48, 0x89, 0xc7]):  # mov rdi, rax
            assembly.append("    mov %rax, %rdi")
            pos += 3
        elif code[pos:pos+4] == bytes([0x48, 0x83, 0xc0]):  # add rax, imm8
            value = code[pos+3]
            assembly.append(f"    add ${value:#x}, %rax")
            pos += 4
        elif code[pos:pos+2] == bytes([0x48, 0x05]):  # add rax, imm32
            value = struct.unpack('<I', code[pos+2:pos+6])[0]
            assembly.append(f"    add ${value:#x}, %rax")
            pos += 6
        elif code[pos:pos+4] == bytes([0x48, 0x83, 0xe8]):  # sub rax, imm8
            value = code[pos+3]
            assembly.append(f"    sub ${value:#x}, %rax")
            pos += 4
        elif code[pos:pos+4] == bytes([0x48, 0x6b, 0xc0]):  # imul rax, imm8
            value = code[pos+3]
            assembly.append(f"    imul ${value:#x}, %rax")
            pos += 4
        elif code[pos:pos+4] == bytes([0x48, 0x8b, 0x05]):  # mov rax, [rip + disp32]
            offset = struct.unpack('<i', code[pos+3:pos+7])[0]
            assembly.append(f"    mov {offset:#x}(%rip), %rax")
            pos += 7
        elif code[pos:pos+4] == bytes([0x48, 0x89, 0x05]):  # mov [rip + disp32], rax
            offset = struct.unpack('<i', code[pos+3:pos+7])[0]
            assembly.append(f"    mov %rax, {offset:#x}(%rip)")
            pos += 7
        elif code[pos:pos+2] == bytes([0x0f, 0x05]):  # syscall
            assembly.append("    syscall")
            pos += 2
        else:
            assembly.append(f"    .byte {code[pos]:#x}  # Unknown instruction")
            pos += 1
    return '\n'.join(assembly)

def compile_to_elf(source: str, output_path: str) -> None:
    """Compile SimpleScript source to ELF executable with assembly output"""
    # Parse to AST
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    
    # Compile to native code
    compiler = NativeCompiler()
    
    # Print AST for debugging
    print("AST:")
    for node in ast:
        print(f"  {node}")
    
    # Compile and show assembly
    compiler.compile_program(ast)
    print("\nGenerated Assembly:")
    print(generate_assembly(compiler.assembler.code))
    
    # Create ELF file
    elf_data = compiler.compile_program(ast)
    
    # Write executable
    with open(output_path, 'wb') as f:
        f.write(elf_data)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"\nCompiled to '{output_path}'. Run with ./{output_path}")

import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any
import os


class AssemblyGenerator:
    def __init__(self):
        self.assembly = []
        self.data_section = []
        self.string_counter = 0
        self.variables: Dict[str, str] = {}
        # Add number format string once at initialization
        self.number_format = self.add_string("%d")
        
    def add_string(self, s: str) -> str:
        """Add string to data section and return its label"""
        label = f"str_{self.string_counter}"
        self.string_counter += 1
        escaped = s.replace('"', '\\"')
        self.data_section.extend([
            f"{label}:",
            f'    db "{escaped}", 0',
            f"{label}_len equ $ - {label}"
        ])
        return label

    def generate_PrintNode(self, node: PrintNode) -> None:
        if isinstance(node.expression, StringNode):
            self.generate_StringNode(node.expression)
        else:
            # For numbers, we use printf but without return
            self.assembly.extend([
                "    ; Print number",
                "    push rdi           ; Save registers",
                "    push rsi",
                "    push rax",
                ])

            # Generate the number into rax
            self.generate_node(node.expression)

            self.assembly.extend([
                "    mov rsi, rax        ; Number to print",
                f"    mov rdi, {self.number_format} ; Format string",
                "    xor rax, rax        ; Clear AL for printf",
                "    call printf",

                "    ; Print newline",
                "    mov rax, 1          ; sys_write",
                "    mov rdi, 1          ; stdout",
                "    push 10             ; newline character",
                "    mov rsi, rsp        ; pointer to newline",
                "    mov rdx, 1          ; length 1",
                "    syscall",
                "    add rsp, 8          ; Clean up newline from stack",

                "    pop rax             ; Restore registers",
                "    pop rsi",
                "    pop rdi"
            ])

    def generate(self, ast: List[Any]) -> str:
        # Generate assembly header
        self.assembly.extend([
            "BITS 64",
            "global _start",
            "extern printf",
            "",
            "section .text",
            "_start:",
            "    ; Setup stack frame",
            "    push rbp",
            "    mov rbp, rsp",
            "    sub rsp, 32         ; Reserve some stack space"
        ])

        # Generate code for each AST node
        for node in ast:
            self.generate_node(node)

        # Add exit syscall with proper stack cleanup
        self.assembly.extend([
            "    ; Restore stack and exit",
            "    mov rsp, rbp        ; Restore stack pointer",
            "    pop rbp             ; Restore base pointer",
            "    mov rax, 60         ; sys_exit",
            "    xor rdi, rdi        ; status 0",
            "    syscall"
        ])

        # Add data section
        if self.data_section:
            self.assembly.extend([
                "",
                "section .data align=8"  # Ensure proper alignment
            ])
            self.assembly.extend(self.data_section)

        return "\n".join(self.assembly)

    def get_variable(self, name: str) -> str:
        """Get or create variable label"""
        if name not in self.variables:
            label = f"var_{name}"
            self.variables[name] = label
            self.data_section.extend([
                f"{label}:",
                "    dq 0"  # 8-byte integer
            ])
        return self.variables[name]
    
    def generate_node(self, node: Any) -> None:
        method_name = f'generate_{type(node).__name__}'
        if not hasattr(self, method_name):
            raise ValueError(f"Don't know how to generate code for {type(node)}")
        getattr(self, method_name)(node)
    
    def generate_StringNode(self, node: StringNode) -> None:
        label = self.add_string(node.value)
        self.assembly.extend([
            f"    ; Print string {node.value!r}",
            "    mov rax, 1          ; sys_write",
            "    mov rdi, 1          ; stdout",
            f"    mov rsi, {label}   ; string address",
            f"    mov rdx, {label}_len ; string length",
            "    syscall"
        ])
    
    def generate_NumberNode(self, node: NumberNode) -> None:
        self.assembly.extend([
            f"    ; Load number {node.value}",
            f"    mov rax, {int(node.value)}"
        ])
    
    def generate_VariableNode(self, node: VariableNode) -> None:
        var_label = self.get_variable(node.name)
        self.assembly.extend([
            f"    ; Load variable {node.name}",
            f"    mov rax, [{var_label}]"
        ])
    
    def generate_AssignmentNode(self, node: AssignmentNode) -> None:
        var_label = self.get_variable(node.name)
        self.assembly.extend([
            f"    ; Assign to variable {node.name}"
        ])
        self.generate_node(node.value)  # Result will be in rax
        self.assembly.extend([
            f"    mov [{var_label}], rax"
        ])
    
    def generate_BinaryOpNode(self, node: BinaryOpNode) -> None:
        self.assembly.extend([
            f"    ; Binary operation {node.operator}"
        ])
        
        # Generate code for left operand
        self.generate_node(node.left)
        # Save left result
        self.assembly.append("    push rax")
        
        # Generate code for right operand
        self.generate_node(node.right)
        # Move right result to rbx
        self.assembly.append("    mov rbx, rax")
        # Restore left result to rax
        self.assembly.append("    pop rax")
        
        # Perform operation
        if node.operator == '+':
            self.assembly.append("    add rax, rbx")
        elif node.operator == '-':
            self.assembly.append("    sub rax, rbx")
        elif node.operator == '*':
            self.assembly.append("    imul rax, rbx")
    
def compile_to_executable(source: str, output_path: str) -> None:
    """Compile source to executable"""
    # Parse to AST
    lexer = Lexer(source)
    parser = Parser(lexer)
    ast = parser.parse()
    
    # Generate assembly
    generator = AssemblyGenerator()
    assembly = generator.generate(ast)
    
    # Print the generated assembly for debugging
    print("Generated Assembly:")
    print(assembly)
    
    # Write assembly to temporary file
    with tempfile.NamedTemporaryFile(suffix='.asm', delete=False) as f:
        f.write(assembly.encode())
        asm_file = f.name
    
    # Assemble and link
    try:
        # Assemble
        obj_file = asm_file[:-4] + '.o'
        subprocess.run(['nasm', '-f', 'elf64', asm_file, '-o', obj_file], check=True)
        
        # Link with C library for printf
        subprocess.run(['ld', '-dynamic-linker', '/lib64/ld-linux-x86-64.so.2',
                       obj_file, '-o', output_path, '-lc'], check=True)
        
    finally:
        # Cleanup temporary files
        os.unlink(asm_file)
        if os.path.exists(obj_file):
            os.unlink(obj_file)

if __name__ == "__main__":
    test_program = '''
    print "Hello, Native World!";
    x = 42;
    print x;
    y = x + 10;
    print y;
    '''
    
    compile_to_executable(test_program, "simplescript")
    print("\nGenerated executable 'simplescript'. Run with ./simplescript")
