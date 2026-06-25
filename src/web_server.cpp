#include "web_server.hpp"
#include "mongoose/mongoose.h"
#include <iostream>
#include <algorithm>

namespace causal_chess {

WebServer::WebServer(int port, const std::string& document_root)
    : port(port), document_root(document_root), running(false), mgr(nullptr) {
}

WebServer::~WebServer() {
    stop();
}

void WebServer::start() {
    if (running) return;
    running = true;

    server_thread = std::thread([this]() {
        mg_log_set(MG_LL_ERROR);
        mgr = new struct mg_mgr();
        mg_mgr_init(mgr);
        mg_wakeup_init(mgr);

        std::string listen_url = "http://0.0.0.0:" + std::to_string(port);
        struct mg_connection* c = mg_http_listen(mgr, listen_url.c_str(), _mongoose_event_handler, this);
        if (c == nullptr) {
            std::cerr << "Web server failed to listen on port " << port << std::endl;
            running = false;
            mg_mgr_free(mgr);
            delete mgr;
            mgr = nullptr;
            return;
        }

        std::cout << "Web server started on " << listen_url << " serving " << document_root << std::endl;

        while (running) {
            mg_mgr_poll(mgr, 50);
        }

        {
            std::lock_guard<std::mutex> lock(connections_mutex);
            connections.clear();
        }

        mg_mgr_free(mgr);
        delete mgr;
        mgr = nullptr;
    });
}

void WebServer::stop() {
    if (!running) return;
    running = false;
    if (server_thread.joinable()) {
        server_thread.join();
    }
}

void WebServer::broadcast(const std::string& message) {
    if (!running || !mgr) return;
    std::lock_guard<std::mutex> lock(connections_mutex);
    for (auto* c : connections) {
        mg_wakeup(mgr, c->id, message.c_str(), message.size());
    }
}

void WebServer::on_message(MessageCallback cb) {
    std::lock_guard<std::mutex> lock(callback_mutex);
    message_callback = cb;
}

void WebServer::_add_connection(struct mg_connection* c) {
    std::lock_guard<std::mutex> lock(connections_mutex);
    connections.push_back(c);
}

void WebServer::_remove_connection(struct mg_connection* c) {
    std::lock_guard<std::mutex> lock(connections_mutex);
    auto it = std::find(connections.begin(), connections.end(), c);
    if (it != connections.end()) {
        connections.erase(it);
    }
}

void WebServer::_mongoose_event_handler(struct mg_connection* c, int ev, void* ev_data) {
    auto* server = static_cast<WebServer*>(c->fn_data);
    if (!server) return;

    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = static_cast<struct mg_http_message*>(ev_data);
        if (mg_match(hm->uri, mg_str("/ws"), NULL)) {
            mg_ws_upgrade(c, hm, NULL);
        } else {
            struct mg_http_serve_opts opts = {0};
            opts.root_dir = server->document_root.c_str();
            mg_http_serve_dir(c, hm, &opts);
        }
    } else if (ev == MG_EV_WS_OPEN) {
        server->_add_connection(c);
    } else if (ev == MG_EV_WS_MSG) {
        struct mg_ws_message* wm = static_cast<struct mg_ws_message*>(ev_data);
        std::string message(wm->data.buf, wm->data.len);
        
        MessageCallback cb;
        {
            std::lock_guard<std::mutex> lock(server->callback_mutex);
            cb = server->message_callback;
        }
        if (cb) {
            cb(message);
        }
    } else if (ev == MG_EV_CLOSE) {
        if (c->is_websocket) {
            server->_remove_connection(c);
        }
    } else if (ev == MG_EV_WAKEUP) {
        struct mg_str* data = static_cast<struct mg_str*>(ev_data);
        if (c->is_websocket) {
            mg_ws_send(c, data->buf, data->len, WEBSOCKET_OP_TEXT);
        }
    }
}

} // namespace causal_chess
