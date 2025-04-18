const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = (env, argv) => {
    const isProd = argv.mode === 'production';
    return {
        entry: './src/script.js',
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'bundle.[contenthash].js',
            publicPath: '',
            clean: true,
        },
        devServer: {
            static: path.resolve(__dirname, 'dist'),
            open: false,
            hot: true,
            port: 3000,
        },
        devtool: isProd ? false : 'eval-source-map',
        plugins: [
            new HtmlWebpackPlugin({
                template: './public/index.html',
            }),
            new CopyWebpackPlugin({
                patterns: [
                    {
                        from: 'public',
                        globOptions: {
                            ignore: ['**/index.html', '**/.DS_Store'],
                        },
                    },
                ],
            }),
        ],
        module: {
            rules: [
            ],
        },
    };
};