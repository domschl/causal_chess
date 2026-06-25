#ifndef CAUSAL_CHESS_WEB_SERVER_HPP
#define CAUSAL_CHESS_WEB_SERVER_HPP

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

// Forward declare mongoose struct
struct mg_mgr;
struct mg_connection;

namespace causal_chess {

class WebServer {
public:
    WebServer(int port = 8080, const std::string& document_root = "web");
    ~WebServer();

    // Start serving HTTP and WS in a background thread
    void start();

    // Stop the server
    void stop();

    // Broadcast a JSON message to all connected WebSocket clients
    void broadcast(const std::string& message);

    // Register callback for when a WebSocket message is received
    using MessageCallback = std::function<void(const std::string&)>;
    void on_message(MessageCallback cb);

    // Internal methods used by callback
    void _add_connection(struct mg_connection* c);
    void _remove_connection(struct mg_connection* c);

private:
    static void _mongoose_event_handler(struct mg_connection* c, int ev, void* ev_data);

    int port;
    std::string document_root;
    std::thread server_thread;
    std::atomic<bool> running;
    
    // Mutex for connection tracking
    std::mutex connections_mutex;
    std::vector<struct mg_connection*> connections;

    std::mutex callback_mutex;
    MessageCallback message_callback;

    // We store pointer/data of mongoose manager
    struct mg_mgr* mgr;
};

} // namespace causal_chess

#endif // CAUSAL_CHESS_WEB_SERVER_HPP
