module.exports = {
    mqtt: {
        port: 3001, //the port where to create the server.
        host: 'localhost', //the IP address of the server
        http: {port: 3002, bundle: true, static: './'}
    }
}

//http://www.mosca.io/docs/lib/server.js.html