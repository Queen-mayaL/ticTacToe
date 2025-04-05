import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
    const [board, setBoard] = useState(Array(9).fill(0));  // Board initialized with 0 (empty)
    const [currentPlayer, setCurrentPlayer] = useState(1); // AI (X) starts first
    const [gameStatus, setGameStatus] = useState("continue");

    // Start the game with the AI making the first move
    useEffect(() => {
        if (currentPlayer === 1) {  // If AI's turn (starts first as X)
            getAIMove(board);
        }
    }, [currentPlayer]);

    const handleClick = (i) => {
        if (gameStatus !== "continue") return;  // If the game is over, no further moves can be made

        const newBoard = [...board];
        if (newBoard[i] === 0 && currentPlayer === -1) {  // Only allow player O (-1) to make moves
            newBoard[i] = -1;  // Player O makes a move
            setBoard(newBoard);
            checkStatus(newBoard);  // Check game status after player's move
        }
    };

    const getAIMove = async (newBoard) => {
        // Ensure the board contains only valid values (0, 1, or -1)
        const validBoard = newBoard.map(val => (val === null ? 0 : val));

        // Check if player O has already won before letting the AI take a move
        const oWins = await checkStatus(newBoard, true); // Add flag to only check O's win status

        if (oWins) {
            setGameStatus("O wins!");  // If O has already won, stop the game
            return;
        }

        try {
            const response = await axios.post("https://tictactoe-1-a87f.onrender.com/get_move", {
                board: validBoard,
            });
            const aiMove = response.data.move;

            const updatedBoard = [...newBoard];
            updatedBoard[aiMove] = 1;  // AI plays as X (1)
            setBoard(updatedBoard);
            await checkStatus(updatedBoard);  // Check game status after AI move
            setCurrentPlayer(-1);  // Switch to player's turn after AI move
        } catch (error) {
            console.error("Error getting AI move:", error);
        }
    };

    const checkStatus = async (newBoard, checkForOWin = false) => {
        try {
            const response = await axios.post("https://tictactoe-1-a87f.onrender.com/check_status", {
                board: newBoard,
                current_player: currentPlayer, // This is the last player who made a move
            });

            const status = response.data.status;

            if (status === "win") {
                const winner = response.data.player === 1 ? "X" : "O"; // Update winner (X or O)
                setGameStatus(`${winner} wins!`);
                return true;  // Stop game, return true if O has won
            } else if (status === "draw") {
                setGameStatus("It's a draw!");
            } else {
                setGameStatus("continue");
                setCurrentPlayer(-currentPlayer); // Switch player
            }

            return false;  // Game continues
        } catch (error) {
            console.error("Error checking game status:", error);
            return false;  // In case of error, continue game
        }
    };

    const renderSquare = (i) => {
        return (
            <td
                key={i}
                className="square"
                onClick={() => handleClick(i)}
                style={{
                    width: "50px",
                    height: "50px",
                    textAlign: "center",
                    border: "1px solid black",
                    cursor: currentPlayer === -1 && board[i] === 0 ? "pointer" : "not-allowed", // Only allow player to click on empty spots
                }}
            >
                {board[i] === 1 ? "X" : board[i] === -1 ? "O" : null}
            </td>
        );
    };

    return (
        <div>
            <table className="board" style={{ margin: "20px auto", borderCollapse: "collapse" }}>
                <tbody>
                    {[...Array(3)].map((_, row) => (
                        <tr key={row}>
                            {[...Array(3)].map((_, col) => renderSquare(row * 3 + col))}
                        </tr>
                    ))}
                </tbody>
            </table>
            <div className="status" style={{ textAlign: 'center' }}>
                <h2>{gameStatus}</h2>
            </div>
        </div>
    );
};

export default App;
